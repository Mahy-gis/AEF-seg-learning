import argparse
from pathlib import Path
from typing import Tuple, List
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.ndimage import zoom


# ---------- Data preparation helpers ----------


def prepare_features_from_embeddings(emb_npz: Path) -> np.ndarray:
    """Load embeddings npz and return features as (C,H,W) float32.

    Supports two formats:
      - embeddings: (H,W,64)
      - embeddings_per_time: (T,H,W,64) -> (T*64,H,W)
    """
    data = np.load(emb_npz)

    if "embeddings_per_time" in data:
        e = data["embeddings_per_time"]  # (T,H,W,64)
        if e.ndim != 4 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings_per_time shape {e.shape} in {emb_npz}")
        e_chw = np.transpose(e, (0, 3, 1, 2))  # (T,64,H,W)
        T, C, H, W = e_chw.shape
        feats = e_chw.reshape(T * C, H, W)
    elif "embeddings" in data:
        e = data["embeddings"]  # (H,W,64)
        if e.ndim != 3 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings shape {e.shape} in {emb_npz}")
        feats = np.transpose(e, (2, 0, 1))  # (64,H,W)
    else:
        raise ValueError(
            f"Neither 'embeddings_per_time' nor 'embeddings' found in {emb_npz}"
        )

    return feats.astype(np.float32)


def resize_labels_to(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Resize label mask (H_l,W_l) to match features (C,H_f,W_f) using nearest neighbor."""
    C, H_f, W_f = features.shape
    H_l, W_l = labels.shape

    if (H_f, W_f) == (H_l, W_l):
        return labels

    zoom_h = H_f / H_l
    zoom_w = W_f / W_l
    resized = zoom(labels, zoom=(zoom_h, zoom_w), order=0)
    return resized.astype(labels.dtype)


class EmbeddingSegmentationDataset(Dataset):
    """Segmentation dataset from AEF embeddings and integer labels.

    Supports two modes:
      - Single-sample: embeddings_path is a file (one npz) -> len=1.
      - Multi-sample: embeddings_path is a directory -> all *.npz inside
        (e.g., embedding_timeseries_*.npz) are treated as separate samples.

    In both cases, labels_npz provides a (H,W) label mask which is resized
    to each sample's spatial resolution as needed.

    Optionally, when per_patch_labels=True, labels_path is treated as a
    directory containing per-patch label files whose names follow the
    pattern ParcelIDs_XXXXX_labels.npz, and embeddings are named
    embedding_XXXXX*.npz (or any name ending with the same numeric XXXXX).
    """
    def __init__(self, embeddings_path: Path, labels_path: Path, per_patch_labels: bool = False):
        self.embeddings_path = embeddings_path
        self.per_patch_labels = per_patch_labels

        if per_patch_labels:
            if not labels_path.is_dir():
                raise ValueError(
                    f"With per_patch_labels=True, labels_path must be a directory, got {labels_path}"
                )
            self.labels_dir = labels_path
            self.labels_np = None
        else:
            # Load a single global label mask once; will be resized per-sample if needed.
            data = np.load(labels_path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile) and "labels" in data:
                labels = data["labels"]
            else:
                labels = data  # assume npy directly

            if labels.ndim != 2:
                raise ValueError(f"Expected labels with shape (H,W), got {labels.shape}")
            self.labels_np = labels
            self.labels_dir = None

        if embeddings_path.is_dir():
            # Collect all npz files under the directory.
            self.files: List[Path] = sorted(
                [p for p in embeddings_path.glob("*.npz") if p.is_file()]
            )
            if not self.files:
                raise FileNotFoundError(
                    f"No .npz files found in embeddings directory {embeddings_path}"
                )
        else:
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings npz not found: {embeddings_path}")
            self.files = [embeddings_path]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_file = self.files[idx]
        feats = prepare_features_from_embeddings(emb_file)  # (C,H,W)

        if self.per_patch_labels:
            # Parse numeric suffix XXXXX from embedding filename (e.g., embedding_XXXXX.npz
            # or any name whose stem ends with digits). Use it to construct
            # the corresponding label filename ParcelIDs_XXXXX_labels.npz.
            stem = emb_file.stem
            m = re.search(r"(\d+)$", stem)
            if m is None:
                raise ValueError(
                    f"Cannot extract numeric patch id from embedding filename {emb_file.name}"
                )
            patch_str = m.group(1)
            label_file = self.labels_dir / f"ParcelIDs_{patch_str}_labels.npz"  # type: ignore[operator]
            if not label_file.exists():
                raise FileNotFoundError(
                    f"Per-patch label file not found for embedding {emb_file.name}: {label_file}"
                )
            data = np.load(label_file, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile) and "labels" in data:
                labels_np = data["labels"]
            else:
                labels_np = data
            if labels_np.ndim != 2:
                raise ValueError(
                    f"Expected labels with shape (H,W) in {label_file}, got {labels_np.shape}"
                )
        else:
            labels_np = self.labels_np  # type: ignore[assignment]

        labels_resized = resize_labels_to(feats, labels_np)
        features = torch.from_numpy(feats)  # (C,H,W)
        labels = torch.from_numpy(labels_resized.astype(np.int64))  # (H,W)
        return features, labels


# ---------- Simple U-Net implementation ----------


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 20, base_ch: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        xb = self.bottleneck(self.pool3(x3))

        # Decoder with skip connections
        x = self.up3(xb)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)


# ---------- Training script ----------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    emb_path = Path(args.embeddings_path)
    labels_path = Path(args.labels_file)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {emb_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    full_dataset = EmbeddingSegmentationDataset(
        emb_path,
        labels_path,
        per_patch_labels=args.per_patch_labels,
    )

    # Peek at one sample to infer channel and spatial sizes.
    sample_feats, sample_labels = full_dataset[0]
    C, H, W = sample_feats.shape
    print(f"Full dataset size: {len(full_dataset)} samples")
    print(f"Sample features shape (C,H,W)=({C},{H},{W}), labels shape (H,W)=({H},{W})")

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_fraction)) if n_total > 1 else 0
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:] if n_val > 0 else np.array([], dtype=int)

    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist()) if n_val > 0 else None

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds is not None else 0}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None
    )

    model = UNet(in_channels=C, num_classes=args.num_classes, base_ch=args.base_channels)
    model.to(device)

    # Void label (19) is treated as ignore_index by default
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B,num_classes,H,W)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                mask = y != args.ignore_index
                correct = (preds[mask] == y[mask]).sum().item()
                total = mask.sum().item()
                running_correct += correct
                running_total += total

        train_loss = running_loss / max(1, len(train_ds))
        train_acc = running_correct / max(1, running_total) if running_total > 0 else 0.0

        # Validation
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_correct = 0
            val_running_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_running_loss += loss.item() * x.size(0)

                    preds = logits.argmax(dim=1)
                    mask = y != args.ignore_index
                    correct = (preds[mask] == y[mask]).sum().item()
                    total = mask.sum().item()
                    val_running_correct += correct
                    val_running_total += total

            val_loss = val_running_loss / max(1, len(val_ds))
            val_acc = (
                val_running_correct / max(1, val_running_total)
                if val_running_total > 0
                else 0.0
            )

            # Track best model by validation loss (then accuracy as tie-breaker)
            improved = val_loss < best_val_loss or (
                np.isclose(val_loss, best_val_loss) and val_acc > best_val_acc
            )
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = model.state_dict()

        if val_loss is not None:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save latest model
    ckpt_latest = out_dir / "unet_from_embeddings_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "in_channels": C}, ckpt_latest)
    print(f"Saved latest U-Net checkpoint to {ckpt_latest}")

    # Save best model (by validation)
    if best_state is not None:
        ckpt_best = out_dir / "unet_from_embeddings_best.pt"
        torch.save({"model_state_dict": best_state, "in_channels": C}, ckpt_best)
        print(
            f"Saved best U-Net checkpoint to {ckpt_best} "
            f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a simple U-Net segmentation model using AEF embeddings "
            "as input features and integer label mask as targets."
        )
    )
    p.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help=(
            "Path to embeddings. If a file, uses a single sample. If a "
            "directory, uses all *.npz files inside as separate samples "
            "(e.g., embedding_timeseries_*.npz)."
        ),
    )
    p.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help=(
            "Labels file (.npz with 'labels' key, or .npy integer mask). "
            "When --per_patch_labels is set, this should instead be a directory "
            "containing per-patch files named ParcelIDs_XXXXX_labels.npz."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs",
    )
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training and validation (default 1)",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=20,
        help="Number of segmentation classes (default 20)",
    )
    p.add_argument(
        "--ignore_index",
        type=int,
        default=19,
        help="Label index to ignore in loss (e.g., 'Void label' = 19)",
    )
    p.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base number of channels in U-Net encoder (default 32)",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of samples to use for validation when multiple "
            "embeddings are provided (default 0.2). Ignored when there "
            "is only one sample."
        ),
    )
    p.add_argument(
        "--per_patch_labels",
        action="store_true",
        help=(
            "If set, treat --labels_file as a directory with per-patch label "
            "files named ParcelIDs_XXXXX_labels.npz that correspond 1:1 to "
            "embedding_XXXXX*.npz files in --embeddings_path."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
