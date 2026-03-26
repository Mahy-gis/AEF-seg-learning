import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from seg.train_unet_from_embeddings import (
    EmbeddingSegmentationDataset,
    FeatureNormalizeDataset,
    compute_segmentation_metrics,
    estimate_feature_channel_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate AEF embedding semantic separability with a linear probe "
            "(single 1x1 conv classifier) and report validation mIoU."
        )
    )
    p.add_argument("--embeddings_path", type=str, required=True, help="Embeddings file or directory")
    p.add_argument("--labels_file", type=str, required=True, help="Label file or per-patch label directory")
    p.add_argument("--embedding_key", type=str, default="embeddings_native", help="Key in npz to read embeddings")
    p.add_argument("--per_patch_labels", action="store_true", help="Force per-patch label matching")
    p.add_argument(
        "--superclass_map_json",
        type=str,
        default="",
        help="Optional JSON path for superclass remapping (dict or list). Mutually exclusive with --binary_mode.",
    )

    p.add_argument("--epochs", type=int, default=20, help="Linear probe training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for linear probe")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    p.add_argument("--num_classes", type=int, default=20, help="Number of classes")
    p.add_argument("--ignore_index", type=int, default=19, help="Ignore label index")
    p.add_argument("--background_index", type=int, default=0, help="Background class index")
    p.add_argument(
        "--binary_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, remap labels to binary (background=0, foreground=1, ignore unchanged)",
    )
    p.add_argument(
        "--ignore_background_in_metrics",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, exclude background class when computing mIoU",
    )

    p.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction")
    p.add_argument(
        "--overfit_single_sample",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, use the same single sample for both train and val (overfit sanity check)",
    )
    p.add_argument(
        "--single_sample_index",
        type=int,
        default=0,
        help="Sample index to use when overfit_single_sample=1 (default 0)",
    )
    p.add_argument(
        "--normalize_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, apply channel-wise feature normalization from train split",
    )
    p.add_argument(
        "--feature_stats_max_samples",
        type=int,
        default=0,
        help="Max samples for feature mean/std estimation (0 means all)",
    )

    p.add_argument("--class_weight_min", type=float, default=0.2, help="Min clamp for class weights")
    p.add_argument("--class_weight_max", type=float, default=5.0, help="Max clamp for class weights")

    p.add_argument("--diagnose_threshold", type=float, default=0.05, help="Diagnostic threshold for best val mIoU")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="cuda/cpu, auto if omitted")
    p.add_argument("--output_dir", type=str, default="", help="Optional output dir to save probe metrics json")
    p.add_argument(
        "--k_sweep",
        type=str,
        default="",
        help="Optional comma-separated K values for train-sample sweep, e.g. '1,4,16,64'",
    )
    p.add_argument(
        "--k_sweep_repeats",
        type=int,
        default=1,
        help="How many random repeats per K in K-sweep (default 1)",
    )
    return p.parse_args()


def parse_k_values(k_text: str) -> list[int]:
    vals: list[int] = []
    for part in (k_text or "").split(","):
        s = part.strip()
        if not s:
            continue
        k = int(s)
        if k <= 0:
            continue
        vals.append(k)
    return sorted(set(vals))


def load_superclass_mapping(
    map_json_path: str,
    num_classes: int,
    ignore_index: int,
) -> tuple[torch.Tensor, int]:
    with open(map_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping = np.arange(num_classes, dtype=np.int64)
    if isinstance(payload, list):
        for i, v in enumerate(payload[:num_classes]):
            mapping[i] = int(v)
    elif isinstance(payload, dict):
        for k, v in payload.items():
            idx = int(k)
            if 0 <= idx < num_classes:
                mapping[idx] = int(v)
    else:
        raise ValueError("Superclass map JSON must be either a list or a dict.")

    if 0 <= ignore_index < num_classes:
        mapping[ignore_index] = ignore_index

    valid_ids = [int(mapping[c]) for c in range(num_classes) if c != ignore_index]
    if any(v < 0 for v in valid_ids):
        raise ValueError("Superclass ids must be >= 0 for non-ignore classes.")

    effective_num_classes = (max(valid_ids) + 1) if valid_ids else 1
    map_tensor = torch.as_tensor(mapping, dtype=torch.long)
    return map_tensor, int(effective_num_classes)


def remap_labels_for_mode(
    y: torch.Tensor,
    binary_mode: bool,
    ignore_index: int,
    background_index: int,
    superclass_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if superclass_map is not None:
        out = y.clone()
        valid = out != ignore_index
        if valid.any():
            mapper = superclass_map.to(out.device)
            out[valid] = mapper[out[valid]]
        return out

    if not binary_mode:
        return y
    ignore = y == ignore_index
    bg = y == background_index
    out = torch.ones_like(y)
    out[bg] = 0
    out[ignore] = ignore_index
    return out


def compute_label_counts(
    ds,
    num_classes: int,
    ignore_index: int,
    binary_mode: bool,
    background_index: int,
    superclass_map: Optional[torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    valid_pixels = 0
    ignore_pixels = 0
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for _x, y in loader:
        y = remap_labels_for_mode(
            y,
            binary_mode=binary_mode,
            ignore_index=ignore_index,
            background_index=background_index,
            superclass_map=superclass_map,
        )
        flat = y.view(-1)
        ignore = flat == ignore_index
        ignore_pixels += int(ignore.sum().item())
        valid = flat[~ignore]
        valid_pixels += int(valid.numel())
        if valid.numel() > 0:
            counts += torch.bincount(valid, minlength=num_classes).to(torch.float64)
    return counts, valid_pixels, ignore_pixels


def class_weights_from_counts(
    counts: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32)
    if present.any():
        total = counts[present].sum()
        weights[present] = (total / counts[present]).to(torch.float32)
        weights[present] = weights[present] / weights[present].mean().clamp_min(1e-6)
        weights[present] = weights[present].clamp(min=min_weight, max=max_weight)
    if 0 <= ignore_index < num_classes:
        weights[ignore_index] = 0.0
    return weights


def export_class_frequency_csv(
    out_file: Path,
    counts: torch.Tensor,
    valid_pixels: int,
    ignore_pixels: int,
) -> None:
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "pixel_count", "pixel_ratio_over_valid", "valid_pixels", "ignore_pixels"])
        denom = max(1, int(valid_pixels))
        for cid in range(int(counts.shape[0])):
            c = float(counts[cid].item())
            writer.writerow([cid, int(round(c)), f"{c/denom:.8f}", int(valid_pixels), int(ignore_pixels)])


def compute_per_class_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> list[Optional[float]]:
    preds = logits.argmax(dim=1)
    mask = target != ignore_index
    out: list[Optional[float]] = []
    for cls in range(num_classes):
        if cls == ignore_index:
            out.append(None)
            continue
        p = (preds == cls) & mask
        t = (target == cls) & mask
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union <= 0:
            out.append(None)
        else:
            out.append(float(inter) / float(union))
    return out


def evaluate_loader(
    probe: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    background_index: int,
    ignore_background_in_metrics: bool,
    binary_mode: bool,
    original_background_index: int,
    superclass_map: Optional[torch.Tensor],
) -> tuple[float, float, float, list[Optional[float]]]:
    if loader is None:
        return 0.0, 0.0, 0.0, [None for _ in range(num_classes)]

    probe.eval()
    val_loss_sum = 0.0
    val_miou_sum = 0.0
    val_miou_n = 0
    val_correct = 0
    val_total = 0
    inter = [0.0 for _ in range(num_classes)]
    union = [0.0 for _ in range(num_classes)]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = remap_labels_for_mode(
                y,
                binary_mode=binary_mode,
                ignore_index=ignore_index,
                background_index=original_background_index,
                superclass_map=superclass_map,
            )

            logits = probe(x)
            loss = criterion(logits, y)
            val_loss_sum += loss.item() * x.size(0)

            acc, miou = compute_segmentation_metrics(
                logits,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
                background_index=background_index,
                ignore_background=ignore_background_in_metrics,
            )
            mask = y != ignore_index
            total = mask.sum().item()
            val_correct += int(round(acc * total))
            val_total += int(total)
            val_miou_sum += miou
            val_miou_n += 1

            batch_pc = compute_per_class_iou(logits, y, num_classes=num_classes, ignore_index=ignore_index)
            for c in range(num_classes):
                if batch_pc[c] is None:
                    continue
                # Recompute intersection/union accumulation exactly.
                preds = logits.argmax(dim=1)
                p = ((preds == c) & mask).sum().item()
                t = ((y == c) & mask).sum().item()
                i = (((preds == c) & (y == c)) & mask).sum().item()
                u = p + t - i
                inter[c] += float(i)
                union[c] += float(u)

    val_loss = val_loss_sum / max(1, len(loader.dataset))
    val_miou = val_miou_sum / max(1, val_miou_n)
    val_acc = val_correct / max(1, val_total)
    per_class: list[Optional[float]] = []
    for c in range(num_classes):
        if c == ignore_index or union[c] <= 0:
            per_class.append(None)
        else:
            per_class.append(inter[c] / union[c])
    return val_loss, val_acc, val_miou, per_class


def run_probe_once(
    dataset: EmbeddingSegmentationDataset,
    args: argparse.Namespace,
    device: torch.device,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    run_label: str = "main",
    superclass_map: Optional[torch.Tensor] = None,
) -> dict:
    sample_x, _ = dataset[0]
    in_channels = int(sample_x.shape[0])
    binary_mode = bool(args.binary_mode)
    if binary_mode:
        effective_num_classes = 2
        effective_background_index = 0
        effective_ignore_background_in_metrics = False
    elif superclass_map is not None:
        map_np = superclass_map.detach().cpu().numpy().astype(np.int64)
        valid = [int(map_np[c]) for c in range(int(args.num_classes)) if c != int(args.ignore_index)]
        effective_num_classes = (max(valid) + 1) if valid else 1
        effective_background_index = int(map_np[int(args.background_index)])
        effective_ignore_background_in_metrics = bool(args.ignore_background_in_metrics)
    else:
        effective_num_classes = int(args.num_classes)
        effective_background_index = int(args.background_index)
        effective_ignore_background_in_metrics = bool(args.ignore_background_in_metrics)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist()) if len(val_idx) > 0 else None

    if args.normalize_features == 1 and len(train_ds) > 0:
        feat_mean, feat_std = estimate_feature_channel_stats(
            train_ds,
            max_samples=args.feature_stats_max_samples,
        )
        train_ds = FeatureNormalizeDataset(train_ds, feat_mean, feat_std)
        if val_ds is not None:
            val_ds = FeatureNormalizeDataset(val_ds, feat_mean, feat_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    probe = nn.Conv2d(in_channels, effective_num_classes, kernel_size=1, bias=True).to(device)

    train_counts, train_valid_pixels, train_ignore_pixels = compute_label_counts(
        train_ds,
        num_classes=effective_num_classes,
        ignore_index=args.ignore_index,
        binary_mode=binary_mode,
        background_index=args.background_index,
        superclass_map=superclass_map,
    )
    val_counts = torch.zeros(effective_num_classes, dtype=torch.float64)
    val_valid_pixels = 0
    val_ignore_pixels = 0
    if val_ds is not None:
        val_counts, val_valid_pixels, val_ignore_pixels = compute_label_counts(
            val_ds,
            num_classes=effective_num_classes,
            ignore_index=args.ignore_index,
            binary_mode=binary_mode,
            background_index=args.background_index,
            superclass_map=superclass_map,
        )

    class_weights = class_weights_from_counts(
        counts=train_counts,
        num_classes=effective_num_classes,
        ignore_index=args.ignore_index,
        min_weight=args.class_weight_min,
        max_weight=args.class_weight_max,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index, weight=class_weights)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_miou = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        probe.train()
        train_loss_sum = 0.0
        train_miou_sum = 0.0
        train_miou_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y = remap_labels_for_mode(
                y,
                binary_mode=binary_mode,
                ignore_index=args.ignore_index,
                background_index=args.background_index,
                superclass_map=superclass_map,
            )

            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            with torch.no_grad():
                _acc, miou = compute_segmentation_metrics(
                    logits,
                    y,
                    num_classes=effective_num_classes,
                    ignore_index=args.ignore_index,
                    background_index=effective_background_index,
                    ignore_background=effective_ignore_background_in_metrics,
                )
                train_miou_sum += miou
                train_miou_n += 1

        train_loss = train_loss_sum / max(1, len(train_loader.dataset))
        train_miou = train_miou_sum / max(1, train_miou_n)

        if val_loader is not None:
            val_loss, val_acc, val_miou, _ = evaluate_loader(
                probe=probe,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=effective_num_classes,
                ignore_index=args.ignore_index,
                background_index=effective_background_index,
                ignore_background_in_metrics=effective_ignore_background_in_metrics,
                binary_mode=binary_mode,
                original_background_index=args.background_index,
                superclass_map=superclass_map,
            )
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
                }

            print(
                f"[{run_label}] Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_mIoU: {train_miou:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mIoU: {val_miou:.4f}, "
                f"best_val_mIoU: {best_val_miou:.4f} (epoch {best_epoch})"
            )
        else:
            if train_miou > best_val_miou:
                best_val_miou = train_miou
                best_val_acc = 0.0
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
                }
            print(
                f"[{run_label}] Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_mIoU: {train_miou:.4f}"
            )

    if best_state is not None:
        probe.load_state_dict(best_state)

    _val_loss, _val_acc, _val_miou, per_class_iou = evaluate_loader(
        probe=probe,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=effective_num_classes,
        ignore_index=args.ignore_index,
        background_index=effective_background_index,
        ignore_background_in_metrics=effective_ignore_background_in_metrics,
        binary_mode=binary_mode,
        original_background_index=args.background_index,
        superclass_map=superclass_map,
    )

    return {
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_val_miou),
        "best_val_acc": float(best_val_acc),
        "embedding_key": args.embedding_key,
        "diagnose_threshold": float(args.diagnose_threshold),
        "binary_mode": int(args.binary_mode),
        "effective_num_classes": int(effective_num_classes),
        "per_class_iou": per_class_iou,
        "train_class_counts": [float(x) for x in train_counts.tolist()],
        "train_valid_pixels": int(train_valid_pixels),
        "train_ignore_pixels": int(train_ignore_pixels),
        "val_class_counts": [float(x) for x in val_counts.tolist()],
        "val_valid_pixels": int(val_valid_pixels),
        "val_ignore_pixels": int(val_ignore_pixels),
    }


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    emb_path = Path(args.embeddings_path)
    labels_path = Path(args.labels_file)
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {emb_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path not found: {labels_path}")

    per_patch_labels = bool(args.per_patch_labels or labels_path.is_dir())

    dataset = EmbeddingSegmentationDataset(
        embeddings_path=emb_path,
        labels_path=labels_path,
        per_patch_labels=per_patch_labels,
        embedding_key=args.embedding_key,
    )

    n_total = len(dataset)
    if n_total <= 0:
        raise ValueError("Empty dataset.")

    if args.binary_mode == 1 and args.superclass_map_json:
        raise ValueError("--binary_mode and --superclass_map_json are mutually exclusive.")

    superclass_map = None
    if args.superclass_map_json:
        map_path = Path(args.superclass_map_json)
        if not map_path.exists():
            raise FileNotFoundError(f"Superclass map JSON not found: {map_path}")
        superclass_map, mapped_classes = load_superclass_mapping(
            map_json_path=str(map_path),
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        print(
            f"Superclass mode: ON ({map_path}, effective_num_classes={mapped_classes})"
        )

    if args.overfit_single_sample == 1:
        sample_idx = int(np.clip(args.single_sample_index, 0, n_total - 1))
        train_idx = np.array([sample_idx], dtype=int)
        val_idx = np.array([sample_idx], dtype=int)
        print(
            f"Overfit mode: ON (single sample index={sample_idx}); "
            "train/val use the same sample."
        )
    else:
        n_val = max(1, int(n_total * args.val_fraction)) if n_total > 1 else 0
        n_train = n_total - n_val
        perm = np.random.permutation(n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:] if n_val > 0 else np.array([], dtype=int)

    result = run_probe_once(
        dataset=dataset,
        args=args,
        device=device,
        train_idx=train_idx,
        val_idx=val_idx,
        run_label="main",
        superclass_map=superclass_map,
    )

    print("\n=== Linear Probe Summary ===")
    print(json.dumps(result, ensure_ascii=True, indent=2))

    if result["best_val_miou"] >= args.diagnose_threshold:
        print(
            "Diagnosis: embedding has moderate/usable class separability under linear probe; "
            "segmentation bottleneck is more likely in decoder/training strategy."
        )
    else:
        print(
            "Diagnosis: linear probe mIoU is low; embedding semantic separability is likely the main bottleneck."
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "linear_probe_metrics.json"
        out_file.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved probe metrics to {out_file}")

        # Export per-class IoU table for the main run.
        per_class_file = out_dir / "linear_probe_per_class_iou.csv"
        with per_class_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "iou"])
            per_class = result.get("per_class_iou", [])
            for cid in range(int(result.get("effective_num_classes", args.num_classes))):
                iou = per_class[cid] if cid < len(per_class) else None
                if iou is None:
                    writer.writerow([cid, ""])
                else:
                    writer.writerow([cid, f"{float(iou):.8f}"])
        print(f"Saved per-class IoU CSV to {per_class_file}")

        # Export class-frequency statistics on current train/val split.
        train_freq_file = out_dir / "linear_probe_train_class_frequency.csv"
        export_class_frequency_csv(
            out_file=train_freq_file,
            counts=torch.as_tensor(result.get("train_class_counts", []), dtype=torch.float64),
            valid_pixels=int(result.get("train_valid_pixels", 0)),
            ignore_pixels=int(result.get("train_ignore_pixels", 0)),
        )
        print(f"Saved train class-frequency CSV to {train_freq_file}")

        val_freq_file = out_dir / "linear_probe_val_class_frequency.csv"
        export_class_frequency_csv(
            out_file=val_freq_file,
            counts=torch.as_tensor(result.get("val_class_counts", []), dtype=torch.float64),
            valid_pixels=int(result.get("val_valid_pixels", 0)),
            ignore_pixels=int(result.get("val_ignore_pixels", 0)),
        )
        print(f"Saved val class-frequency CSV to {val_freq_file}")

        # Optional K-sample sweep: write table for curve plotting.
        k_values = parse_k_values(args.k_sweep)
        if k_values:
            print(f"\n=== K-sweep start: {k_values}, repeats={args.k_sweep_repeats} ===")
            rows = []
            base_train_idx = train_idx.copy()
            sweep_rng = np.random.RandomState(args.seed)
            for k in k_values:
                k_eff = min(k, len(base_train_idx))
                if k_eff <= 0:
                    continue
                for rep in range(max(1, int(args.k_sweep_repeats))):
                    if args.overfit_single_sample == 1:
                        k_train_idx = base_train_idx[:1]
                    else:
                        perm_local = sweep_rng.permutation(len(base_train_idx))
                        k_train_idx = base_train_idx[perm_local[:k_eff]]

                    k_result = run_probe_once(
                        dataset=dataset,
                        args=args,
                        device=device,
                        train_idx=k_train_idx,
                        val_idx=val_idx,
                        run_label=f"k={k_eff},rep={rep+1}",
                        superclass_map=superclass_map,
                    )
                    rows.append(
                        {
                            "k": int(k_eff),
                            "repeat": int(rep + 1),
                            "train_samples": int(len(k_train_idx)),
                            "val_samples": int(len(val_idx)),
                            "best_epoch": int(k_result["best_epoch"]),
                            "best_val_miou": float(k_result["best_val_miou"]),
                            "best_val_acc": float(k_result["best_val_acc"]),
                        }
                    )

            table_file = out_dir / "linear_probe_k_sweep_table.csv"
            with table_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "k",
                        "repeat",
                        "train_samples",
                        "val_samples",
                        "best_epoch",
                        "best_val_miou",
                        "best_val_acc",
                    ],
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Saved K-sweep table CSV to {table_file}")

            # Aggregate means for direct curve plotting.
            curve_file = out_dir / "linear_probe_k_curve.csv"
            by_k: dict[int, list[dict]] = {}
            for r in rows:
                by_k.setdefault(int(r["k"]), []).append(r)
            with curve_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["k", "n_runs", "mean_best_val_miou", "std_best_val_miou", "mean_best_val_acc", "std_best_val_acc"])
                for k in sorted(by_k.keys()):
                    rs = by_k[k]
                    miou_vals = np.array([float(x["best_val_miou"]) for x in rs], dtype=np.float64)
                    acc_vals = np.array([float(x["best_val_acc"]) for x in rs], dtype=np.float64)
                    writer.writerow([
                        k,
                        len(rs),
                        f"{miou_vals.mean():.8f}",
                        f"{miou_vals.std(ddof=0):.8f}",
                        f"{acc_vals.mean():.8f}",
                        f"{acc_vals.std(ddof=0):.8f}",
                    ])
            print(f"Saved K-curve CSV to {curve_file}")


if __name__ == "__main__":
    main()
