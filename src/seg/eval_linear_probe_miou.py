import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from seg.train_unet_from_embeddings import (
    EmbeddingSegmentationDataset,
    FeatureNormalizeDataset,
    compute_segmentation_metrics,
    estimate_feature_channel_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate AEF embedding semantic separability with a pixel-wise probe "
            "(linear 1x1 conv or shallow MLP) and report validation mIoU."
        )
    )
    p.add_argument("--embeddings_path", type=str, required=True, help="Embeddings file or directory")
    p.add_argument("--labels_file", type=str, required=True, help="Label file or per-patch label directory")
    p.add_argument(
        "--val_embeddings_path",
        type=str,
        default="",
        help="Optional explicit validation embeddings path; if set, disables random split",
    )
    p.add_argument(
        "--val_labels_file",
        type=str,
        default="",
        help="Optional explicit validation labels path; must be set together with --val_embeddings_path",
    )
    p.add_argument("--embedding_key", type=str, default="embeddings_native", help="Key in npz to read embeddings")
    p.add_argument("--per_patch_labels", action="store_true", help="Force per-patch label matching")
    p.add_argument(
        "--superclass_map_json",
        type=str,
        default="",
        help="Optional JSON path for superclass remapping (dict or list). Mutually exclusive with --binary_mode.",
    )

    p.add_argument("--epochs", type=int, default=40, help="Probe training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for probe")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument(
        "--probe_head",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Pixel classifier head type: linear=single 1x1 conv, mlp=1x1-MLP (Linear->ReLU->Linear)",
    )
    p.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=128,
        help="Hidden channels for MLP probe head when --probe_head=mlp",
    )
    p.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside MLP probe head",
    )
    p.add_argument(
        "--ce_weight",
        type=float,
        default=0.6,
        help="Weight for CE term in mixed CE+Focal loss",
    )
    p.add_argument(
        "--focal_weight",
        type=float,
        default=0.4,
        help="Weight for Focal term in mixed CE+Focal loss",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma in focal loss",
    )

    p.add_argument("--num_classes", type=int, default=9, help="Number of classes")
    p.add_argument("--ignore_index", type=int, default=0, help="Ignore label index")
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
    p.add_argument(
        "--boost_classes",
        type=str,
        default="4,6",
        help="Comma-separated class ids to further upweight after inverse-frequency weighting",
    )
    p.add_argument(
        "--boost_factor",
        type=float,
        default=1.5,
        help="Multiplicative factor for --boost_classes weights",
    )
    p.add_argument(
        "--boost_max_weight",
        type=float,
        default=20.0,
        help="Upper bound after applying --boost_factor to target classes",
    )
    p.add_argument(
        "--sample_boost_factor",
        type=float,
        default=3.0,
        help="Per-sample multiplier when a patch contains boosted classes",
    )
    p.add_argument(
        "--sample_boost_min_pixels",
        type=int,
        default=64,
        help="Minimum number of target-class pixels to mark a sample as boosted",
    )

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
    p.add_argument(
        "--save_val_predictions",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, save per-sample val probability map and segmentation map from the best probe",
    )
    return p.parse_args()


class PixelMLPProbe(nn.Module):
    """Per-pixel MLP classifier implemented with 1x1 convolutions.

    This is equivalent to applying Linear -> ReLU -> Linear independently to
    every pixel embedding vector.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = max(1, int(hidden_dim))
        p = float(np.clip(dropout, 0.0, 1.0))
        # 3-layer pixel MLP: in->hidden->hidden->classes with residual path.
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.fc3 = nn.Conv2d(hidden, num_classes, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p) if p > 0.0 else nn.Identity()
        self.res_proj = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res_proj(x)
        h = self.act(self.bn1(self.fc1(x)))
        h = self.act(self.bn2(self.fc2(h) + res))
        h = self.drop(h)
        return self.fc3(h)


def focal_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    gamma: float,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Ignore masked pixels first, then compute focal over valid pixels only.
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    t = target[valid]
    # logits: (N,C,H,W) -> (num_valid, C)
    logp = logits.permute(0, 2, 3, 1)[valid]
    log_prob = torch.log_softmax(logp, dim=1)
    prob = torch.exp(log_prob)
    pt = prob.gather(1, t.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
    ce = -log_prob.gather(1, t.unsqueeze(1)).squeeze(1)
    focal = ((1.0 - pt) ** float(gamma)) * ce

    if alpha is not None:
        a = alpha.to(logits.device, dtype=logits.dtype)
        focal = focal * a[t]

    return focal.mean()


def compute_mixed_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_criterion: nn.Module,
    ignore_index: int,
    ce_weight: float,
    focal_weight: float,
    focal_gamma: float,
    focal_alpha: Optional[torch.Tensor],
) -> torch.Tensor:
    ce = ce_criterion(logits, target)
    fl = focal_loss_multiclass(
        logits=logits,
        target=target,
        ignore_index=ignore_index,
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    w_ce = float(max(0.0, ce_weight))
    w_fl = float(max(0.0, focal_weight))
    denom = max(1e-8, w_ce + w_fl)
    return (w_ce * ce + w_fl * fl) / denom


def build_probe_head(
    in_channels: int,
    num_classes: int,
    head_type: str,
    mlp_hidden_dim: int,
    mlp_dropout: float,
) -> nn.Module:
    if head_type == "linear":
        return nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
    if head_type == "mlp":
        return PixelMLPProbe(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
        )
    raise ValueError(f"Unsupported probe head type: {head_type}")


def save_val_predictions(
    probe: nn.Module,
    ds,
    out_dir: Path,
    device: torch.device,
    binary_mode: bool,
    ignore_index: int,
    background_index: int,
    superclass_map: Optional[torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    probe.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y = remap_labels_for_mode(
                y,
                binary_mode=binary_mode,
                ignore_index=ignore_index,
                background_index=background_index,
                superclass_map=superclass_map,
            )

            logits = probe(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            save_path = out_dir / f"val_pred_{i:05d}.npz"
            np.savez_compressed(
                save_path,
                probs=probs[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32),
                pred=pred[0].detach().cpu().numpy().astype(np.int64),
                target=y[0].detach().cpu().numpy().astype(np.int64),
            )


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


def parse_class_list(text: str) -> list[int]:
    vals: list[int] = []
    for part in (text or "").split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(int(s))
    return sorted(set(vals))


def build_sample_weights_for_boosted_classes(
    dataset: Dataset,
    train_idx: np.ndarray,
    boost_classes: list[int],
    ignore_index: int,
    min_pixels: int,
    sample_boost_factor: float,
) -> tuple[Optional[torch.Tensor], int]:
    if len(train_idx) == 0 or not boost_classes or sample_boost_factor <= 1.0:
        return None, 0

    min_pixels = max(1, int(min_pixels))
    weights = torch.ones(len(train_idx), dtype=torch.float32)
    boosted = 0

    for i, ds_idx in enumerate(train_idx.tolist()):
        _x, y = dataset[int(ds_idx)]
        flat = y.reshape(-1)
        valid = flat[flat != ignore_index]
        if valid.numel() == 0:
            continue

        has_target = False
        for cls in boost_classes:
            if int((valid == cls).sum().item()) >= min_pixels:
                has_target = True
                break

        if has_target:
            weights[i] = float(sample_boost_factor)
            boosted += 1

    return weights, boosted


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
        flat = y.reshape(-1)
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


def export_confusion_matrix_csv(
    out_file: Path,
    confusion: list[list[int]],
) -> None:
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not confusion:
            writer.writerow(["target\\pred"])
            return
        num_classes = len(confusion)
        writer.writerow(["target\\pred"] + [str(i) for i in range(num_classes)])
        for i, row in enumerate(confusion):
            writer.writerow([str(i)] + [int(x) for x in row])


def export_confusion_matrix_png(
    out_file: Path,
    confusion: list[list[int]],
    title: str,
) -> bool:
    if not confusion:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    cm = np.asarray(confusion, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[1] == 0:
        return False

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=np.float64)
    np.divide(cm, np.maximum(row_sums, 1.0), out=cm_norm, where=row_sums > 0)

    n = int(cm.shape[0])
    fig_size = max(6.0, min(14.0, 0.75 * n + 2.0))
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2.0, fig_size), dpi=150)

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("Target class")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{title} (row-normalized)")
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("Target class")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return True


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
    ce_criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    background_index: int,
    ignore_background_in_metrics: bool,
    binary_mode: bool,
    original_background_index: int,
    superclass_map: Optional[torch.Tensor],
    ce_weight: float,
    focal_weight: float,
    focal_gamma: float,
    focal_alpha: Optional[torch.Tensor],
) -> tuple[float, float, float, float, float, list[Optional[float]], list[Optional[float]], list[list[int]]]:
    if loader is None:
        empty_conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        return 0.0, 0.0, 0.0, 0.0, 0.0, [None for _ in range(num_classes)], [None for _ in range(num_classes)], empty_conf

    probe.eval()
    val_loss_sum = 0.0
    val_miou_sum = 0.0
    val_miou_n = 0
    val_correct = 0
    val_total = 0
    inter = [0.0 for _ in range(num_classes)]
    union = [0.0 for _ in range(num_classes)]
    pred_pixels = [0.0 for _ in range(num_classes)]
    target_pixels = [0.0 for _ in range(num_classes)]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

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
            loss = compute_mixed_loss(
                logits=logits,
                target=y,
                ce_criterion=ce_criterion,
                ignore_index=ignore_index,
                ce_weight=ce_weight,
                focal_weight=focal_weight,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
            val_loss_sum += loss.item() * x.size(0)

            acc, miou = compute_segmentation_metrics(
                logits,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
                background_index=background_index,
                ignore_background=ignore_background_in_metrics,
            )
            preds = logits.argmax(dim=1)
            mask = y != ignore_index
            total = mask.sum().item()
            val_correct += int(round(acc * total))
            val_total += int(total)
            val_miou_sum += miou
            val_miou_n += 1

            t = y[mask].reshape(-1)
            p = preds[mask].reshape(-1)
            if t.numel() > 0:
                idx = (t * num_classes + p).detach().cpu()
                binc = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
                confusion += binc.numpy().astype(np.int64)

            batch_pc = compute_per_class_iou(logits, y, num_classes=num_classes, ignore_index=ignore_index)
            for c in range(num_classes):
                if batch_pc[c] is None:
                    continue
                # Recompute intersection/union accumulation exactly.
                p = ((preds == c) & mask).sum().item()
                t = ((y == c) & mask).sum().item()
                i = (((preds == c) & (y == c)) & mask).sum().item()
                u = p + t - i
                inter[c] += float(i)
                union[c] += float(u)
                pred_pixels[c] += float(p)
                target_pixels[c] += float(t)

    val_loss = val_loss_sum / max(1, len(loader.dataset))
    val_miou = val_miou_sum / max(1, val_miou_n)
    val_acc = val_correct / max(1, val_total)
    per_class: list[Optional[float]] = []
    per_class_f1: list[Optional[float]] = []
    for c in range(num_classes):
        if c == ignore_index or union[c] <= 0:
            per_class.append(None)
            per_class_f1.append(None)
        else:
            per_class.append(inter[c] / union[c])
            precision = inter[c] / max(1e-12, pred_pixels[c])
            recall = inter[c] / max(1e-12, target_pixels[c])
            denom = precision + recall
            if denom <= 0:
                per_class_f1.append(0.0)
            else:
                per_class_f1.append(2.0 * precision * recall / denom)

    # Keep F1 class filtering aligned with metric setting (optionally ignore background).
    f1_vals: list[float] = []
    f1_weighted_num = 0.0
    f1_weighted_den = 0.0
    for c, f1 in enumerate(per_class_f1):
        if f1 is None:
            continue
        if ignore_background_in_metrics and c == background_index:
            continue
        f1_vals.append(float(f1))
        f1_weighted_num += float(f1) * float(target_pixels[c])
        f1_weighted_den += float(target_pixels[c])
    val_f1_macro = float(np.mean(f1_vals)) if f1_vals else 0.0
    val_f1_weighted = (f1_weighted_num / f1_weighted_den) if f1_weighted_den > 0 else 0.0

    return (
        val_loss,
        val_acc,
        val_miou,
        val_f1_macro,
        val_f1_weighted,
        per_class,
        per_class_f1,
        confusion.tolist(),
    )


def run_probe_once(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    run_label: str = "main",
    superclass_map: Optional[torch.Tensor] = None,
    return_probe: bool = False,
) -> dict | tuple[dict, nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
    feat_mean: Optional[torch.Tensor] = None
    feat_std: Optional[torch.Tensor] = None
    boosted_classes = parse_class_list(getattr(args, "boost_classes", ""))
    train_sample_weights, n_boosted_samples = build_sample_weights_for_boosted_classes(
        dataset=dataset,
        train_idx=train_idx,
        boost_classes=boosted_classes,
        ignore_index=args.ignore_index,
        min_pixels=getattr(args, "sample_boost_min_pixels", 64),
        sample_boost_factor=getattr(args, "sample_boost_factor", 1.0),
    )

    if args.normalize_features == 1 and len(train_ds) > 0:
        feat_mean, feat_std = estimate_feature_channel_stats(
            train_ds,
            max_samples=args.feature_stats_max_samples,
        )
        train_ds = FeatureNormalizeDataset(train_ds, feat_mean, feat_std)
        if val_ds is not None:
            val_ds = FeatureNormalizeDataset(val_ds, feat_mean, feat_std)

    train_sampler = None
    if train_sample_weights is not None and len(train_sample_weights) == len(train_ds):
        train_sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    probe = build_probe_head(
        in_channels=in_channels,
        num_classes=effective_num_classes,
        head_type=args.probe_head,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
    ).to(device)

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

    boost_factor = float(getattr(args, "boost_factor", 1.0))
    if boosted_classes and boost_factor > 0.0:
        max_w = float(getattr(args, "boost_max_weight", 20.0))
        for c in boosted_classes:
            if 0 <= c < effective_num_classes and c != int(args.ignore_index):
                class_weights[c] = torch.clamp(class_weights[c] * boost_factor, max=max_w)

    ce_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index, weight=class_weights)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_miou = -1.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_f1_weighted = 0.0
    best_val_confusion: Optional[list[list[int]]] = None
    best_epoch = 0
    best_state = None

    focal_alpha = class_weights.detach().to(device=device, dtype=torch.float32)

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
            loss = compute_mixed_loss(
                logits=logits,
                target=y,
                ce_criterion=ce_criterion,
                ignore_index=args.ignore_index,
                ce_weight=args.ce_weight,
                focal_weight=args.focal_weight,
                focal_gamma=args.focal_gamma,
                focal_alpha=focal_alpha,
            )
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
            val_loss, val_acc, val_miou, val_f1, val_f1_weighted, _pc_iou, _pc_f1, val_confusion = evaluate_loader(
                probe=probe,
                loader=val_loader,
                ce_criterion=ce_criterion,
                device=device,
                num_classes=effective_num_classes,
                ignore_index=args.ignore_index,
                background_index=effective_background_index,
                ignore_background_in_metrics=effective_ignore_background_in_metrics,
                binary_mode=binary_mode,
                original_background_index=args.background_index,
                superclass_map=superclass_map,
                ce_weight=args.ce_weight,
                focal_weight=args.focal_weight,
                focal_gamma=args.focal_gamma,
                focal_alpha=focal_alpha,
            )
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_f1_weighted = val_f1_weighted
                best_val_confusion = val_confusion
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
                }

            print(
                f"[{run_label}] Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_mIoU: {train_miou:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_F1_macro: {val_f1:.4f}, "
                f"val_F1_weighted: {val_f1_weighted:.4f}, val_mIoU: {val_miou:.4f}, "
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

    _val_loss, _val_acc, _val_miou, final_val_f1, final_val_f1_weighted, per_class_iou, per_class_f1, final_val_confusion = evaluate_loader(
        probe=probe,
        loader=val_loader,
        ce_criterion=ce_criterion,
        device=device,
        num_classes=effective_num_classes,
        ignore_index=args.ignore_index,
        background_index=effective_background_index,
        ignore_background_in_metrics=effective_ignore_background_in_metrics,
        binary_mode=binary_mode,
        original_background_index=args.background_index,
        superclass_map=superclass_map,
        ce_weight=args.ce_weight,
        focal_weight=args.focal_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
    )

    result = {
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_val_miou),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "best_val_f1_weighted": float(best_val_f1_weighted),
        "final_val_f1": float(final_val_f1),
        "final_val_f1_weighted": float(final_val_f1_weighted),
        "embedding_key": args.embedding_key,
        "probe_head": str(args.probe_head),
        "mlp_hidden_dim": int(args.mlp_hidden_dim),
        "mlp_dropout": float(args.mlp_dropout),
        "ce_weight": float(args.ce_weight),
        "focal_weight": float(args.focal_weight),
        "focal_gamma": float(args.focal_gamma),
        "diagnose_threshold": float(args.diagnose_threshold),
        "binary_mode": int(args.binary_mode),
        "effective_num_classes": int(effective_num_classes),
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
        "best_val_confusion_matrix": best_val_confusion if best_val_confusion is not None else final_val_confusion,
        "final_val_confusion_matrix": final_val_confusion,
        "train_class_counts": [float(x) for x in train_counts.tolist()],
        "train_valid_pixels": int(train_valid_pixels),
        "train_ignore_pixels": int(train_ignore_pixels),
        "val_class_counts": [float(x) for x in val_counts.tolist()],
        "val_valid_pixels": int(val_valid_pixels),
        "val_ignore_pixels": int(val_ignore_pixels),
        "class_weights": [float(x) for x in class_weights.detach().cpu().tolist()],
        "boost_classes": boosted_classes,
        "boost_factor": float(boost_factor),
        "boost_max_weight": float(getattr(args, "boost_max_weight", 20.0)),
        "sample_boost_factor": float(getattr(args, "sample_boost_factor", 1.0)),
        "sample_boost_min_pixels": int(getattr(args, "sample_boost_min_pixels", 64)),
        "boosted_train_samples": int(n_boosted_samples),
        "train_samples": int(len(train_idx)),
        "in_channels": int(in_channels),
    }
    if return_probe:
        return result, probe, feat_mean, feat_std
    return result


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

    use_explicit_val = bool(args.val_embeddings_path or args.val_labels_file)
    if bool(args.val_embeddings_path) != bool(args.val_labels_file):
        raise ValueError("--val_embeddings_path and --val_labels_file must be provided together.")

    per_patch_labels = bool(args.per_patch_labels or labels_path.is_dir())

    if use_explicit_val:
        val_emb_path = Path(args.val_embeddings_path)
        val_labels_path = Path(args.val_labels_file)
        if not val_emb_path.exists():
            raise FileNotFoundError(f"Val embeddings path not found: {val_emb_path}")
        if not val_labels_path.exists():
            raise FileNotFoundError(f"Val labels path not found: {val_labels_path}")

        val_per_patch_labels = bool(args.per_patch_labels or val_labels_path.is_dir())
        train_dataset = EmbeddingSegmentationDataset(
            embeddings_path=emb_path,
            labels_path=labels_path,
            per_patch_labels=per_patch_labels,
            embedding_key=args.embedding_key,
        )
        val_dataset = EmbeddingSegmentationDataset(
            embeddings_path=val_emb_path,
            labels_path=val_labels_path,
            per_patch_labels=val_per_patch_labels,
            embedding_key=args.embedding_key,
        )
        if len(train_dataset) <= 0:
            raise ValueError("Empty train dataset.")
        if len(val_dataset) <= 0:
            raise ValueError("Empty val dataset.")

        dataset = ConcatDataset([train_dataset, val_dataset])
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        train_idx = np.arange(0, n_train, dtype=int)
        val_idx = np.arange(n_train, n_train + n_val, dtype=int)
        print(
            "Using explicit train/val datasets: "
            f"train_samples={n_train}, val_samples={n_val}"
        )
    else:
        dataset = EmbeddingSegmentationDataset(
            embeddings_path=emb_path,
            labels_path=labels_path,
            per_patch_labels=per_patch_labels,
            embedding_key=args.embedding_key,
        )

        n_total = len(dataset)
        if n_total <= 0:
            raise ValueError("Empty dataset.")

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

    if use_explicit_val and args.overfit_single_sample == 1:
        sample_idx = int(np.clip(args.single_sample_index, 0, len(train_idx) - 1))
        train_idx = np.array([train_idx[sample_idx]], dtype=int)
        val_idx = np.array([train_idx[sample_idx]], dtype=int)
        print(
            f"Overfit mode with explicit val: ON (train sample index={sample_idx}); "
            "train/val use the same train sample."
        )

    result, trained_probe, feat_mean, feat_std = run_probe_once(
        dataset=dataset,
        args=args,
        device=device,
        train_idx=train_idx,
        val_idx=val_idx,
        run_label="main",
        superclass_map=superclass_map,
        return_probe=True,
    )

    print("\n=== Pixel Probe Summary ===")
    print(json.dumps(result, ensure_ascii=True, indent=2))

    if result["best_val_miou"] >= args.diagnose_threshold:
        print(
            "Diagnosis: embedding has moderate/usable class separability under the selected probe; "
            "segmentation bottleneck is more likely in decoder/training strategy."
        )
    else:
        print(
            "Diagnosis: probe mIoU is low; embedding semantic separability is likely the main bottleneck."
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

        best_conf_file = out_dir / "linear_probe_best_val_confusion_matrix.csv"
        export_confusion_matrix_csv(
            out_file=best_conf_file,
            confusion=result.get("best_val_confusion_matrix", []),
        )
        print(f"Saved best-val confusion matrix CSV to {best_conf_file}")

        final_conf_file = out_dir / "linear_probe_final_val_confusion_matrix.csv"
        export_confusion_matrix_csv(
            out_file=final_conf_file,
            confusion=result.get("final_val_confusion_matrix", []),
        )
        print(f"Saved final-val confusion matrix CSV to {final_conf_file}")

        best_conf_png = out_dir / "linear_probe_best_val_confusion_matrix.png"
        if export_confusion_matrix_png(
            out_file=best_conf_png,
            confusion=result.get("best_val_confusion_matrix", []),
            title="Best Val Confusion Matrix",
        ):
            print(f"Saved best-val confusion matrix PNG to {best_conf_png}")
        else:
            print("Skipped best-val confusion matrix PNG export (matplotlib missing or matrix empty).")

        final_conf_png = out_dir / "linear_probe_final_val_confusion_matrix.png"
        if export_confusion_matrix_png(
            out_file=final_conf_png,
            confusion=result.get("final_val_confusion_matrix", []),
            title="Final Val Confusion Matrix",
        ):
            print(f"Saved final-val confusion matrix PNG to {final_conf_png}")
        else:
            print("Skipped final-val confusion matrix PNG export (matplotlib missing or matrix empty).")

        if args.save_val_predictions == 1 and len(val_idx) > 0:
            pred_dir = out_dir / "val_predictions"
            val_ds_for_export = Subset(dataset, val_idx.tolist())
            if args.normalize_features == 1 and len(train_idx) > 0:
                train_ds_for_stats = Subset(dataset, train_idx.tolist())
                feat_mean, feat_std = estimate_feature_channel_stats(
                    train_ds_for_stats,
                    max_samples=args.feature_stats_max_samples,
                )
                val_ds_for_export = FeatureNormalizeDataset(val_ds_for_export, feat_mean, feat_std)

            if feat_mean is not None and feat_std is not None:
                val_ds_for_export = FeatureNormalizeDataset(val_ds_for_export, feat_mean, feat_std)
            save_val_predictions(
                probe=trained_probe,
                ds=val_ds_for_export,
                out_dir=pred_dir,
                device=device,
                binary_mode=bool(args.binary_mode),
                ignore_index=args.ignore_index,
                background_index=args.background_index,
                superclass_map=superclass_map,
            )
            print(f"Saved val prediction NPZ files to {pred_dir}")

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
