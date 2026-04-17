import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg import train_unet_from_embeddings as base


def _infer_checkpoint_meta(state: Dict[str, torch.Tensor]) -> Tuple[str, int, int, int]:
    """Infer (model_variant, depth, base_channels, num_classes) from state_dict keys."""
    keys = set(state.keys())

    if any(k.startswith("aspp.") for k in keys):
        variant = "aspp"
        depth = 4
    elif any(k.startswith("enc1.block1.conv1") for k in keys):
        variant = "resse"
        depth = 5 if any(k.startswith("enc5.") for k in keys) else 4
    else:
        variant = "basic"
        depth = 5 if any(k.startswith("enc5.") for k in keys) else 4

    if variant == "basic":
        base_ch = int(state["enc1.net.0.weight"].shape[0])
    elif variant == "resse":
        base_ch = int(state["enc1.block1.conv1.weight"].shape[0])
    else:
        base_ch = int(state["stem.conv.weight"].shape[0])

    num_classes = int(state["out_conv.weight"].shape[0])
    return variant, depth, base_ch, num_classes


def _build_model(
    variant: str,
    depth: int,
    in_channels: int,
    num_classes: int,
    base_channels: int,
    norm_type: str,
    dropout_rate: float,
) -> torch.nn.Module:
    if variant == "aspp":
        return base.DeepLabLite(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
        )
    if variant == "resse":
        return base.UNetResSE(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
            depth=depth,
            dropout=dropout_rate,
        )
    if depth == 5:
        return base.UNetDeep(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
        )
    return base.UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_channels,
        norm=norm_type,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a UNet-from-embeddings checkpoint on a validation dataset."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")

    p.add_argument(
        "--val_embeddings_path",
        type=str,
        required=True,
        help="Validation embeddings path (file or directory)",
    )
    p.add_argument(
        "--val_labels_file",
        type=str,
        required=True,
        help="Validation labels file (.npy/.npz) or labels directory",
    )

    p.add_argument(
        "--embedding_key",
        type=str,
        default="embeddings_native",
        help="Embedding key in npz to use (auto/embeddings/embeddings_native/embeddings_per_time)",
    )
    p.add_argument(
        "--resample_size",
        type=int,
        default=0,
        help="If >0, resize each embedding/label sample to this square size before evaluation",
    )
    p.add_argument(
        "--per_patch_labels",
        action="store_true",
        help="Treat labels path as a per-patch label directory",
    )

    p.add_argument(
        "--normalize_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, apply train-set channel-wise normalization to val features",
    )
    p.add_argument(
        "--train_embeddings_path",
        type=str,
        default="",
        help="Train embeddings path used to estimate feature normalization stats",
    )
    p.add_argument(
        "--train_labels_file",
        type=str,
        default="",
        help="Train labels path used to estimate feature normalization stats",
    )
    p.add_argument(
        "--feature_stats_max_samples",
        type=int,
        default=0,
        help="Max train samples for estimating mean/std (0 means all)",
    )

    p.add_argument("--batch_size", type=int, default=4, help="Validation batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    p.add_argument(
        "--resume_strict",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, strict state_dict loading",
    )
    p.add_argument(
        "--model_variant",
        type=str,
        default="auto",
        choices=["auto", "basic", "resse", "aspp"],
        help="Model variant override; auto infers from checkpoint",
    )
    p.add_argument(
        "--unet_depth",
        type=int,
        default=0,
        choices=[0, 4, 5],
        help="UNet depth override; 0 means auto infer",
    )
    p.add_argument(
        "--base_channels",
        type=int,
        default=0,
        help="Base channels override; 0 means auto infer",
    )
    p.add_argument(
        "--norm_type",
        type=str,
        default="group",
        choices=["group", "batch"],
        help="Normalization layer type",
    )
    p.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout used by ResSE blocks",
    )

    p.add_argument("--ignore_index", type=int, default=255, help="Ignore label index")
    p.add_argument("--background_index", type=int, default=0, help="Background class index")
    p.add_argument(
        "--ignore_background_in_metrics",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, exclude background in mIoU",
    )
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument(
        "--dice_weight",
        type=float,
        default=0.4,
        help="Total loss = focal + dice_weight * dice",
    )

    p.add_argument(
        "--save_per_class_iou_npy",
        type=str,
        default="",
        help="Optional output .npy path for per-class IoU",
    )
    p.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save evaluation summary JSON",
    )

    return p.parse_args(argv)


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    val_emb_path = Path(args.val_embeddings_path)
    val_labels_path = Path(args.val_labels_file)
    if not val_emb_path.exists():
        raise FileNotFoundError(f"Validation embeddings path not found: {val_emb_path}")
    if not val_labels_path.exists():
        raise FileNotFoundError(f"Validation labels file/path not found: {val_labels_path}")

    per_patch_val = bool(args.per_patch_labels or val_labels_path.is_dir())
    val_ds_raw = base.EmbeddingSegmentationDataset(
        embeddings_path=val_emb_path,
        labels_path=val_labels_path,
        per_patch_labels=per_patch_val,
        embedding_key=args.embedding_key,
        resample_size=args.resample_size,
    )

    if len(val_ds_raw) == 0:
        raise ValueError("Validation dataset is empty")

    val_ds = val_ds_raw
    feat_mean = None
    feat_std = None
    if args.normalize_features == 1:
        if not args.train_embeddings_path or not args.train_labels_file:
            raise ValueError(
                "--normalize_features=1 requires --train_embeddings_path and --train_labels_file"
            )

        train_emb_path = Path(args.train_embeddings_path)
        train_labels_path = Path(args.train_labels_file)
        if not train_emb_path.exists():
            raise FileNotFoundError(f"Train embeddings path not found: {train_emb_path}")
        if not train_labels_path.exists():
            raise FileNotFoundError(f"Train labels file/path not found: {train_labels_path}")

        per_patch_train = bool(args.per_patch_labels or train_labels_path.is_dir())
        train_ds_raw = base.EmbeddingSegmentationDataset(
            embeddings_path=train_emb_path,
            labels_path=train_labels_path,
            per_patch_labels=per_patch_train,
            embedding_key=args.embedding_key,
            resample_size=args.resample_size,
        )
        feat_mean, feat_std = base.estimate_feature_channel_stats(
            train_ds_raw,
            max_samples=args.feature_stats_max_samples,
        )
        val_ds = base.FeatureNormalizeDataset(val_ds_raw, feat_mean, feat_std)

    sample_x, _ = val_ds[0]
    in_channels = int(sample_x.shape[0])

    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    state = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt

    auto_variant, auto_depth, auto_base_ch, num_classes = _infer_checkpoint_meta(state)
    model_variant = auto_variant if args.model_variant == "auto" else args.model_variant
    unet_depth = auto_depth if args.unet_depth == 0 else int(args.unet_depth)
    base_channels = auto_base_ch if args.base_channels == 0 else int(args.base_channels)

    model = _build_model(
        variant=model_variant,
        depth=unet_depth,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        norm_type=args.norm_type,
        dropout_rate=args.dropout_rate,
    )

    model.to(device)
    load_info = model.load_state_dict(state, strict=bool(args.resume_strict))

    if isinstance(load_info, tuple):
        # Older torch compatibility.
        missing_keys, unexpected_keys = load_info
    else:
        missing_keys = getattr(load_info, "missing_keys", [])
        unexpected_keys = getattr(load_info, "unexpected_keys", [])

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    focus_classes = [
        c for c in range(num_classes) if c not in (args.background_index, args.ignore_index)
    ]
    if not focus_classes:
        focus_classes = [c for c in range(num_classes) if c != args.ignore_index]

    criterion_focal = base.FocalLossWithSampling(
        num_classes=num_classes,
        gamma=args.focal_gamma,
        ignore_index=args.ignore_index,
        focus_classes=focus_classes,
        class_weights=None,
    )
    criterion_dice = base.SoftDiceLoss(
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
        ignore_background=True,
    )

    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    val_running_total = 0
    val_running_miou_sum = 0.0
    val_running_miou_count = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logits, y = base.align_logits_and_target_spatial(logits, y)

            loss_focal = criterion_focal(logits, y)
            loss_dice = criterion_dice(logits, y)
            loss = loss_focal + args.dice_weight * loss_dice
            val_running_loss += loss.item() * x.size(0)

            acc, miou = base.compute_segmentation_metrics(
                logits,
                y,
                num_classes=num_classes,
                ignore_index=args.ignore_index,
                background_index=args.background_index,
                ignore_background=bool(args.ignore_background_in_metrics),
            )
            mask = y != args.ignore_index
            total = int(mask.sum().item())
            val_running_correct += int(round(acc * total))
            val_running_total += total
            val_running_miou_sum += float(miou)
            val_running_miou_count += 1

    val_loss = val_running_loss / max(1, len(val_ds))
    val_acc = val_running_correct / max(1, val_running_total) if val_running_total > 0 else 0.0
    val_miou = val_running_miou_sum / max(1, val_running_miou_count)

    per_class_iou = base.evaluate_per_class_iou(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
        ignore_background=bool(args.ignore_background_in_metrics),
    )

    if args.save_per_class_iou_npy:
        out_npy = Path(args.save_per_class_iou_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, per_class_iou)

    result = {
        "checkpoint": str(ckpt_path),
        "embedding_key": args.embedding_key,
        "model_variant": model_variant,
        "unet_depth": int(unet_depth),
        "base_channels": int(base_channels),
        "num_classes": int(num_classes),
        "val_samples": int(len(val_ds)),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_miou": float(val_miou),
        "missing_keys": int(len(missing_keys)),
        "unexpected_keys": int(len(unexpected_keys)),
    }

    if feat_mean is not None and feat_std is not None:
        result["feature_norm_mean_min"] = float(feat_mean.min().item())
        result["feature_norm_mean_max"] = float(feat_mean.max().item())
        result["feature_norm_std_min"] = float(feat_std.min().item())
        result["feature_norm_std_max"] = float(feat_std.max().item())

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print("EVAL_SUMMARY")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
