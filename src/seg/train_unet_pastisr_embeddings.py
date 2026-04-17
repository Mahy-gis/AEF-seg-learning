import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from seg.train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        load_unet_checkpoint,
        prepare_features_from_embeddings,
        resize_features_to,
        resize_labels_to,
        save_val_visualizations,
    )
except ModuleNotFoundError:
    from train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        load_unet_checkpoint,
        prepare_features_from_embeddings,
        resize_features_to,
        resize_labels_to,
        save_val_visualizations,
    )


def _extract_patch_id(path: Path) -> Optional[str]:
    name = path.name
    patterns = [
        r"^embedding_(\d+)\.(npz|npy)$",
        r"^ParcelIDs_(\d+)_labels\.npz$",
        r"^ParcelIDs_(\d+)\.npy$",
        r"^sample_(\d+)\.npz$",
    ]
    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return str(int(m.group(1)))
    return None


def _resolve_split_dir(root: Path, split: str) -> Path:
    split_dir = root / split
    if split_dir.is_dir():
        return split_dir
    return root


def _build_id_file_map(files: List[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for f in files:
        pid = _extract_patch_id(f)
        if pid is not None:
            out[pid] = f
    return out


class PastisREmbeddingDataset(Dataset):
    """Pastis-R dataset using embedding + label files, optionally constrained by raw split IDs.

    Expected naming:
      - embedding: embedding_{ID}.npz (or .npy)
      - label: ParcelIDs_{ID}_labels.npz (or ParcelIDs_{ID}.npy)
      - raw: sample_{ID}.npz (used only for split filtering)
    """

    def __init__(
        self,
        embeddings_root: Path,
        labels_root: Path,
        split: str,
        raw_root: Optional[Path] = None,
        embedding_key: str = "auto",
        label_key: str = "labels",
        resample_size: int = 0,
        require_raw_match: bool = True,
    ):
        self.embedding_key = embedding_key
        self.label_key = label_key
        self.resample_size = max(0, int(resample_size))

        emb_dir = _resolve_split_dir(embeddings_root, split)
        lbl_dir = _resolve_split_dir(labels_root, split)

        if not emb_dir.exists():
            raise FileNotFoundError(f"Embedding split directory not found: {emb_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Label split directory not found: {lbl_dir}")

        emb_files = sorted(
            [
                p
                for ext in ("*.npz", "*.npy")
                for p in emb_dir.glob(ext)
                if p.is_file()
            ]
        )
        lbl_files = sorted(
            [
                p
                for ext in ("*.npz", "*.npy")
                for p in lbl_dir.glob(ext)
                if p.is_file()
            ]
        )

        if not emb_files:
            raise FileNotFoundError(f"No embedding files found in {emb_dir}")
        if not lbl_files:
            raise FileNotFoundError(f"No label files found in {lbl_dir}")

        emb_map = _build_id_file_map(emb_files)
        lbl_map = _build_id_file_map(lbl_files)

        if not emb_map:
            raise RuntimeError(f"No valid embedding IDs parsed in {emb_dir}")
        if not lbl_map:
            raise RuntimeError(f"No valid label IDs parsed in {lbl_dir}")

        common_ids = set(emb_map.keys()) & set(lbl_map.keys())

        if raw_root is not None and require_raw_match:
            raw_dir = _resolve_split_dir(raw_root, split)
            if not raw_dir.exists():
                raise FileNotFoundError(f"Raw split directory not found: {raw_dir}")
            raw_files = sorted([p for p in raw_dir.glob("sample_*.npz") if p.is_file()])
            if not raw_files:
                raise FileNotFoundError(f"No raw sample files found in {raw_dir}")
            raw_map = _build_id_file_map(raw_files)
            common_ids = common_ids & set(raw_map.keys())

        if not common_ids:
            raise RuntimeError(
                "No matched IDs across embedding/label"
                + ("/raw" if (raw_root is not None and require_raw_match) else "")
                + f" for split='{split}'."
            )

        ordered_ids = sorted(common_ids, key=lambda x: int(x))
        self.pairs: List[Tuple[Path, Path, str]] = [
            (emb_map[i], lbl_map[i], i)
            for i in ordered_ids
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_file, lbl_file, _patch_id = self.pairs[idx]

        x_chw = prepare_features_from_embeddings(emb_file, embedding_key=self.embedding_key)

        data = np.load(lbl_file, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile) and self.label_key in data:
            y = data[self.label_key]
        elif isinstance(data, np.lib.npyio.NpzFile) and "labels" in data:
            y = data["labels"]
        else:
            y = data

        y = np.asarray(y).squeeze()
        if y.ndim != 2:
            raise ValueError(f"Expected 2D label in {lbl_file}, got shape {y.shape}")

        y = y.astype(np.int64)

        if self.resample_size > 0:
            x_chw = resize_features_to(x_chw, self.resample_size, self.resample_size)

        y = resize_labels_to(x_chw, y)
        return torch.from_numpy(x_chw), torch.from_numpy(y.astype(np.int64))


def train(args: argparse.Namespace):
    run_start_ts = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Using device: cuda:{dev_idx} ({torch.cuda.get_device_name(dev_idx)})")
    else:
        print("Using device: cpu")

    embeddings_root = Path(args.embeddings_root)
    if args.embeddings_subdir:
        embeddings_root = embeddings_root / args.embeddings_subdir
    labels_root = Path(args.labels_root)
    raw_root = Path(args.raw_root) if args.raw_root else None
    if bool(args.require_raw_match) and raw_root is None:
        raise ValueError("--raw_root is required when --require_raw_match=1")

    train_base_ds = PastisREmbeddingDataset(
        embeddings_root=embeddings_root,
        labels_root=labels_root,
        split=args.train_split,
        raw_root=raw_root,
        embedding_key=args.embedding_key,
        label_key=args.label_key,
        resample_size=args.resample_size,
        require_raw_match=bool(args.require_raw_match),
    )

    val_base_ds = PastisREmbeddingDataset(
        embeddings_root=embeddings_root,
        labels_root=labels_root,
        split=args.val_split,
        raw_root=raw_root,
        embedding_key=args.embedding_key,
        label_key=args.label_key,
        resample_size=args.resample_size,
        require_raw_match=bool(args.require_raw_match),
    )

    sample_feats, sample_labels = train_base_ds[0]
    c, h, w = sample_feats.shape
    print(f"Train split size: {len(train_base_ds)}, Val split size: {len(val_base_ds)}")
    print(f"Sample features (C,H,W)=({c},{h},{w}), labels shape={tuple(sample_labels.shape)}")

    valid_ratio, y_min, y_max = estimate_label_valid_ratio(
        train_base_ds,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_samples=32,
    )
    print(
        f"Label sanity: valid_ratio={valid_ratio:.4f}, min={y_min}, max={y_max}, "
        f"num_classes={args.num_classes}, ignore_index={args.ignore_index}"
    )
    if valid_ratio < args.min_valid_ratio:
        raise ValueError(
            "Label sanity check failed: too many label pixels are outside valid class range. "
            f"valid_ratio={valid_ratio:.4f} < min_valid_ratio={args.min_valid_ratio:.4f}."
        )

    base_train_ds = train_base_ds
    base_val_ds = val_base_ds

    if args.normalize_features == 1 and len(base_train_ds) > 0:
        feat_mean, feat_std = estimate_feature_channel_stats(
            base_train_ds,
            max_samples=args.feature_stats_max_samples,
        )
        print(
            "Feature normalization: ON "
            f"(mean range [{feat_mean.min().item():.4f}, {feat_mean.max().item():.4f}], "
            f"std range [{feat_std.min().item():.4f}, {feat_std.max().item():.4f}])"
        )
        base_train_ds = FeatureNormalizeDataset(base_train_ds, feat_mean, feat_std)
        base_val_ds = FeatureNormalizeDataset(base_val_ds, feat_mean, feat_std)
    else:
        print("Feature normalization: OFF")

    train_ds = base_train_ds
    val_ds = base_val_ds

    if args.enable_augmentation == 1 and len(base_train_ds) > 0:
        train_ds = AugmentedTrainDataset(
            base_dataset=base_train_ds,
            repeat_factor=args.train_repeat_factor,
            flip_prob=args.aug_flip_prob,
            rot90_prob=args.aug_rot90_prob,
            noise_std=args.aug_noise_std,
            gain_std=args.aug_gain_std,
            crop_size=args.train_crop_size,
            foreground_crop_prob=args.fg_crop_prob,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_sampler = None
    if args.use_fg_sampler == 1 and len(base_train_ds) > 0:
        fg_ratios = estimate_sample_foreground_ratios(
            base_train_ds,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )
        weights = args.fg_sampler_min_weight + np.power(
            np.clip(fg_ratios, 1e-6, None),
            args.fg_sampler_power,
        )
        if len(train_ds) != len(base_train_ds):
            rep = int(np.ceil(len(train_ds) / len(base_train_ds)))
            weights = np.tile(weights, rep)[: len(train_ds)]

        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        print("Foreground sampler: ON")
    else:
        print("Foreground sampler: OFF")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.model_variant == "resse":
        model = UNetResSE(
            in_channels=c,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
            depth=args.unet_depth,
            dropout=args.dropout_rate,
        )
    elif args.model_variant == "aspp":
        model = DeepLabLite(
            in_channels=c,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
        )
    else:
        model = (
            UNetDeep(in_channels=c, num_classes=args.num_classes, base_ch=args.base_channels, norm=args.norm_type)
            if args.unet_depth == 5
            else UNet(in_channels=c, num_classes=args.num_classes, base_ch=args.base_channels, norm=args.norm_type)
        )
    model.to(device)

    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        load_unet_checkpoint(model, checkpoint_path=ckpt_path, device=device, strict=bool(args.resume_strict))

    focal_class_weights = None
    if str(args.focal_class_weights).strip():
        focal_class_weights = [
            float(x)
            for x in str(args.focal_class_weights).split(",")
            if x.strip() != ""
        ]

    focus_spec = str(args.focal_focus_classes).strip().lower()
    if focus_spec in ("", "all", "all_fg"):
        focus_classes = [
            cls for cls in range(args.num_classes) if cls not in (args.background_index, args.ignore_index)
        ]
    else:
        parsed = []
        for tok in str(args.focal_focus_classes).split(","):
            tok = tok.strip()
            if tok:
                try:
                    parsed.append(int(tok))
                except ValueError:
                    pass
        focus_classes = [cls for cls in parsed if 0 <= cls < args.num_classes and cls != args.ignore_index]
    if not focus_classes:
        focus_classes = [cls for cls in range(args.num_classes) if cls != args.ignore_index]

    criterion_focal = FocalLossWithSampling(
        num_classes=args.num_classes,
        gamma=args.focal_gamma,
        ignore_index=args.ignore_index,
        focus_classes=focus_classes,
        class_weights=focal_class_weights,
    )
    criterion_dice = SoftDiceLoss(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
        ignore_background=args.ignore_background_in_dice,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_mode = "max" if args.scheduler_monitor == "val_miou" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_miou = 0.0
    best_state = None
    epochs_no_improve = 0
    epochs_ran = 0
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_train_miou = float("nan")
    last_train_pred_bg_ratio = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")
    last_val_miou = float("nan")
    last_val_pred_bg_ratio = float("nan")

    for epoch in range(args.epochs):
        epochs_ran = epoch + 1
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        running_miou_sum = 0.0
        running_miou_count = 0
        running_pred_bg = 0
        running_pred_total = 0

        train_steps = max(1, len(train_loader))
        epoch_start_t = time.time()

        for step_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            logits, y = align_logits_and_target_spatial(logits, y)
            loss_focal = criterion_focal(logits, y)
            loss_dice = criterion_dice(logits, y)
            loss = loss_focal + args.dice_weight * loss_dice
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            with torch.no_grad():
                acc, miou = compute_segmentation_metrics(
                    logits,
                    y,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    background_index=args.background_index,
                    ignore_background=args.ignore_background_in_metrics,
                )
                mask = y != args.ignore_index
                total = mask.sum().item()
                running_correct += int(round(acc * total))
                running_total += int(total)
                pred = logits.argmax(dim=1)
                running_pred_bg += int(((pred == args.background_index) & mask).sum().item())
                running_pred_total += int(total)
                running_miou_sum += miou
                running_miou_count += 1

            if args.log_interval > 0 and (step_idx % args.log_interval == 0 or step_idx == train_steps):
                seen_samples = step_idx * x.size(0)
                avg_loss_so_far = running_loss / max(1, seen_samples)
                avg_miou_so_far = running_miou_sum / max(1, running_miou_count)
                elapsed = time.time() - epoch_start_t
                print(
                    f"Epoch {epoch+1}/{args.epochs} [train {step_idx}/{train_steps}] "
                    f"loss={avg_loss_so_far:.4f}, focal={loss_focal.item():.4f}, dice={loss_dice.item():.4f}, "
                    f"mIoU={avg_miou_so_far:.4f}, elapsed={elapsed:.1f}s"
                )

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = running_correct / max(1, running_total) if running_total > 0 else 0.0
        train_miou = running_miou_sum / max(1, running_miou_count)
        train_pred_bg_ratio = running_pred_bg / max(1, running_pred_total)

        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0
        val_running_miou_sum = 0.0
        val_running_miou_count = 0
        val_running_pred_bg = 0
        val_running_pred_total = 0
        val_steps = max(1, len(val_loader))
        val_start_t = time.time()

        with torch.no_grad():
            for val_step_idx, (x, y) in enumerate(val_loader, start=1):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                logits, y = align_logits_and_target_spatial(logits, y)
                loss_focal = criterion_focal(logits, y)
                loss_dice = criterion_dice(logits, y)
                loss = loss_focal + args.dice_weight * loss_dice
                val_running_loss += loss.item() * x.size(0)

                acc, miou = compute_segmentation_metrics(
                    logits,
                    y,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    background_index=args.background_index,
                    ignore_background=args.ignore_background_in_metrics,
                )
                mask = y != args.ignore_index
                total = mask.sum().item()
                val_running_correct += int(round(acc * total))
                val_running_total += int(total)
                pred = logits.argmax(dim=1)
                val_running_pred_bg += int(((pred == args.background_index) & mask).sum().item())
                val_running_pred_total += int(total)
                val_running_miou_sum += miou
                val_running_miou_count += 1

                if args.log_interval > 0 and (val_step_idx % args.log_interval == 0 or val_step_idx == val_steps):
                    seen_val_samples = val_step_idx * x.size(0)
                    avg_val_loss_so_far = val_running_loss / max(1, seen_val_samples)
                    avg_val_miou_so_far = val_running_miou_sum / max(1, val_running_miou_count)
                    val_elapsed = time.time() - val_start_t
                    print(
                        f"Epoch {epoch+1}/{args.epochs} [val {val_step_idx}/{val_steps}] "
                        f"loss={avg_val_loss_so_far:.4f}, mIoU={avg_val_miou_so_far:.4f}, "
                        f"elapsed={val_elapsed:.1f}s"
                    )

        val_loss = val_running_loss / max(1, len(val_ds))
        val_acc = val_running_correct / max(1, val_running_total) if val_running_total > 0 else 0.0
        val_miou = val_running_miou_sum / max(1, val_running_miou_count)
        val_pred_bg_ratio = val_running_pred_bg / max(1, val_running_pred_total)

        last_train_loss = float(train_loss)
        last_train_acc = float(train_acc)
        last_train_miou = float(train_miou)
        last_train_pred_bg_ratio = float(train_pred_bg_ratio)
        last_val_loss = float(val_loss)
        last_val_acc = float(val_acc)
        last_val_miou = float(val_miou)
        last_val_pred_bg_ratio = float(val_pred_bg_ratio)

        improved = (val_miou > best_val_miou) or (np.isclose(val_miou, best_val_miou) and val_loss < best_val_loss)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_miou = val_miou
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
            f"train_pred_bg: {train_pred_bg_ratio:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mIoU: {val_miou:.4f}, "
            f"val_pred_bg: {val_pred_bg_ratio:.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_miou if args.scheduler_monitor == "val_miou" else val_loss)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(no mIoU improvement for {args.early_stop_patience} epoch(s))."
            )
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_latest = out_dir / "unet_pastisr_embeddings_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "in_channels": c}, ckpt_latest)
    print(f"Saved latest checkpoint to {ckpt_latest}")

    if best_state is not None:
        ckpt_best = out_dir / "unet_pastisr_embeddings_best.pt"
        torch.save({"model_state_dict": best_state, "in_channels": c}, ckpt_best)
        print(
            f"Saved best checkpoint to {ckpt_best} "
            f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, val_mIoU={best_val_miou:.4f})"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        per_class_iou = evaluate_per_class_iou(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            background_index=args.background_index,
            ignore_background=args.ignore_background_in_metrics,
        )
        np.save(out_dir / "best_val_per_class_iou.npy", per_class_iou)

    save_val_visualizations(
        model=model,
        val_ds=val_ds,
        device=device,
        output_dir=out_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.save_val_samples,
    )

    run_summary = {
        "script": "train_unet_pastisr_embeddings.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": float(time.time() - run_start_ts),
        "output_dir": str(out_dir),
        "epochs_target": int(args.epochs),
        "epochs_ran": int(epochs_ran),
        "stopped_early": bool(epochs_ran < args.epochs),
        "best_metrics": {
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "best_val_miou": float(best_val_miou),
        },
        "last_metrics": {
            "train_loss": last_train_loss,
            "train_acc": last_train_acc,
            "train_miou": last_train_miou,
            "train_pred_bg_ratio": last_train_pred_bg_ratio,
            "val_loss": last_val_loss,
            "val_acc": last_val_acc,
            "val_miou": last_val_miou,
            "val_pred_bg_ratio": last_val_pred_bg_ratio,
        },
        "model": {
            "model_variant": args.model_variant,
            "unet_depth": int(args.unet_depth),
            "base_channels": int(args.base_channels),
            "dropout_rate": float(args.dropout_rate),
            "norm_type": args.norm_type,
            "in_channels": int(c),
            "num_classes": int(args.num_classes),
        },
        "optimization": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "dice_weight": float(args.dice_weight),
            "focal_gamma": float(args.focal_gamma),
            "scheduler_monitor": args.scheduler_monitor,
        },
        "labels": {
            "ignore_index": int(args.ignore_index),
            "background_index": int(args.background_index),
            "ignore_background_in_dice": int(args.ignore_background_in_dice),
            "ignore_background_in_metrics": int(args.ignore_background_in_metrics),
            "focal_focus_classes": str(args.focal_focus_classes),
        },
        "data": {
            "embeddings_root": str(embeddings_root),
            "labels_root": str(labels_root),
            "raw_root": str(raw_root) if raw_root is not None else "",
            "train_split": args.train_split,
            "val_split": args.val_split,
            "resample_size": int(args.resample_size),
            "normalize_features": int(args.normalize_features),
            "enable_augmentation": int(args.enable_augmentation),
            "use_fg_sampler": int(args.use_fg_sampler),
        },
        "args": vars(args),
    }
    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved run summary to {summary_path}")

    return {
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_miou": float(best_val_miou),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train and validate U-Net segmentation on Pastis-R embeddings + labels. "
            "Split is controlled by train/val subdirectories and optionally checked against raw sample IDs."
        )
    )

    p.add_argument("--embeddings_root", type=str, default="/mnt/data/mhy/RSFM/AEF-seg-learning/data/Pastis-R/embedding")
    p.add_argument(
        "--embeddings_subdir",
        type=str,
        default="AEF_npz",
        help="Subdirectory under embeddings_root. Set empty string to use embeddings_root directly.",
    )
    p.add_argument("--labels_root", type=str, default="/mnt/data/mhy/RSFM/AEF-seg-learning/data/Pastis-R/label")
    p.add_argument(
        "--raw_root",
        type=str,
        default="",
        help="Optional raw root used only for ID consistency check when --require_raw_match=1.",
    )
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument(
        "--require_raw_match",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, enforce intersection with raw sample IDs for each split.",
    )

    p.add_argument("--embedding_key", type=str, default="auto")
    p.add_argument("--label_key", type=str, default="labels")
    p.add_argument("--resample_size", type=int, default=0)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--ignore_index", type=int, default=19)
    p.add_argument("--background_index", type=int, default=0)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--unet_depth", type=int, default=4, choices=[4, 5])
    p.add_argument("--model_variant", type=str, default="basic", choices=["basic", "resse", "aspp"])
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--norm_type", type=str, default="group", choices=["group", "batch"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume_checkpoint", type=str, default="")
    p.add_argument("--resume_strict", type=int, default=1, choices=[0, 1])
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--enable_augmentation", type=int, default=1, choices=[0, 1])
    p.add_argument("--normalize_features", type=int, default=1, choices=[0, 1])
    p.add_argument("--feature_stats_max_samples", type=int, default=0)
    p.add_argument("--train_repeat_factor", type=int, default=3)
    p.add_argument("--aug_flip_prob", type=float, default=0.5)
    p.add_argument("--aug_rot90_prob", type=float, default=0.5)
    p.add_argument("--aug_noise_std", type=float, default=0.01)
    p.add_argument("--aug_gain_std", type=float, default=0.05)
    p.add_argument("--train_crop_size", type=int, default=96)
    p.add_argument("--fg_crop_prob", type=float, default=0.8)

    p.add_argument("--use_fg_sampler", type=int, default=1, choices=[0, 1])
    p.add_argument("--fg_sampler_power", type=float, default=1.5)
    p.add_argument("--fg_sampler_min_weight", type=float, default=0.2)

    p.add_argument("--dice_weight", type=float, default=0.4)
    p.add_argument("--ignore_background_in_dice", type=int, default=1, choices=[0, 1])
    p.add_argument("--ignore_background_in_metrics", type=int, default=1, choices=[0, 1])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_class_weights", type=str, default="")
    p.add_argument("--focal_focus_classes", type=str, default="all_fg")

    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--scheduler_monitor", type=str, default="val_miou", choices=["val_loss", "val_miou"])
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--save_val_samples", type=int, default=4)
    p.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print intermediate train/val progress every N steps (<=0 disables)",
    )
    p.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.95,
        help="Fail-fast threshold for label sanity check valid ratio.",
    )
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
