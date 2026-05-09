import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
        UNetLightweight,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_confusion_matrix,
        compute_f1_from_confusion_matrix,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        dump_per_class_iou_report,
        load_unet_checkpoint,
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
        UNetLightweight,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_confusion_matrix,
        compute_f1_from_confusion_matrix,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        dump_per_class_iou_report,
        load_unet_checkpoint,
        resize_features_to,
        resize_labels_to,
        save_val_visualizations,
    )


def _normalize_patch_id(raw_id: str) -> str:
    raw_id = str(raw_id)
    return str(int(raw_id)) if raw_id.isdigit() else raw_id


def _extract_patch_id(path: Path) -> Optional[str]:
    name = path.name
    patterns = [
        r"^sample_(\d+)\.(npz|npy)$",
        r"^sample_(\d+)_label\.(npz|npy)$",
        r"^ParcelIDs_(\d+)_labels\.(npz|npy)$",
        r"^ParcelIDs_(\d+)\.npy$",
        r"^(.*)_image\.(npz|npy)$",
        r"^(.*)_label\.(npz|npy)$",
    ]
    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return _normalize_patch_id(m.group(1))
    return None


def _resolve_split_dir(root: Path, split: str) -> Path:
    split_dir = root / split
    if split_dir.is_dir():
        return split_dir
    return root


def _resolve_label_split_dir(root: Path, split: str) -> Path:
    # Prefer common label folder names like train_labels_npz/val_labels_npz.
    candidates = [
        root / f"{split}_labels_npz",
        root / f"{split}_labels",
        root / split,
        root,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return root


def _build_id_file_map(files: List[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for file_path in files:
        patch_id = _extract_patch_id(file_path)
        if patch_id is not None:
            prev = out.get(patch_id)
            if prev is None:
                out[patch_id] = file_path
            else:
                # Prefer .npy when both .npz and .npy exist for the same patch id.
                if prev.suffix.lower() == ".npz" and file_path.suffix.lower() == ".npy":
                    out[patch_id] = file_path
    return out


def _sort_patch_id(patch_id: str) -> Tuple[int, str]:
    if patch_id.isdigit():
        return (0, f"{int(patch_id):09d}")
    return (1, patch_id)


def _to_4d(arr: np.ndarray, key: str, path: Path) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array for key '{key}' in {path}, got shape {arr.shape}")
    return arr


def _to_thwc(arr: np.ndarray, key: str, path: Path) -> np.ndarray:
    """Coerce a 4D array to (T,H,W,C).

    Supported common layouts:
    - (T,H,W,C) (already)
    - (T,C,H,W) (transpose)
    - (C,T,H,W) (transpose)
    """
    x = _to_4d(arr, key=key, path=path)
    t, a, b, c = x.shape

    # Heuristic: spatial dimensions are typically the two largest dims.
    # - If last dim is small (<=64) and middle dims are large -> already THWC.
    # - If second dim is small and last two dims are large -> TCHW.
    # - If first two dims are small and last two dims are large -> assume CTHW.
    if c <= 64 and a > 64 and b > 64:
        return x
    if a <= 64 and b > 64 and c > 64:
        # (T,C,H,W) most common for raw tensors saved from PyTorch.
        # If ambiguous with (C,T,H,W), assume first dim is time when it's >= second dim.
        if t >= a:
            return np.transpose(x, (0, 2, 3, 1))
        return np.transpose(x, (1, 2, 3, 0))
    if t <= 64 and a <= 64 and b > 64 and c > 64:
        # Likely (C,T,H,W)
        return np.transpose(x, (1, 2, 3, 0))

    # Fallback: treat as THWC to avoid silently permuting unexpected shapes.
    return x


def _normalize_source(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    max_abs = float(np.nanmax(np.abs(x))) if x.size > 0 else 0.0
    # Sentinel-2 is often int reflectance scaled by 1e4; normalize to a stable range.
    if max_abs > 100.0:
        x = x / 10000.0
    return x


def _select_and_pad_time(arr: np.ndarray, time_steps: int) -> np.ndarray:
    t, h, w, c = arr.shape
    if time_steps <= 0:
        return arr

    if t >= time_steps:
        return arr[-time_steps:]

    pad = np.zeros((time_steps - t, h, w, c), dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def _flatten_time_channels(arr: np.ndarray) -> np.ndarray:
    # (T,H,W,C) -> (T*C,H,W)
    t, h, w, c = arr.shape
    chw = np.transpose(arr, (0, 3, 1, 2))
    return chw.reshape(t * c, h, w).astype(np.float32)


def _parse_focus_classes(
    focus_spec: str,
    num_classes: int,
    background_index: int,
    ignore_index: int,
) -> List[int]:
    raw = str(focus_spec).strip().lower()
    if raw in ("", "all_fg"):
        return [c for c in range(num_classes) if c not in (background_index, ignore_index)]
    if raw == "all":
        return [c for c in range(num_classes) if c != ignore_index]

    include: Set[int] = set()
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    for tok in tokens:
        if tok == "all_fg":
            include.update(c for c in range(num_classes) if c not in (background_index, ignore_index))
            continue
        if tok == "all":
            include.update(c for c in range(num_classes) if c != ignore_index)
            continue
        if tok in ("bg", "background"):
            if 0 <= background_index < num_classes and background_index != ignore_index:
                include.add(background_index)
            continue
        try:
            idx = int(tok)
        except ValueError:
            continue
        if 0 <= idx < num_classes and idx != ignore_index:
            include.add(idx)

    if not include:
        include.update(c for c in range(num_classes) if c != ignore_index)
    return sorted(include)


def _metric_class_indices(
    num_classes: int,
    background_index: int,
    ignore_index: int,
    ignore_background: bool,
) -> List[int]:
    class_indices = [c for c in range(num_classes) if c != ignore_index]
    if ignore_background:
        class_indices = [c for c in class_indices if c != background_index]
    return class_indices


class PastisRawSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        label_root: Path,
        split: str,
        s1_key: str = "sentinel1",
        s2_key: str = "sentinel2",
        label_key: str = "labels",
        time_steps: int = 12,
        resample_size: int = 0,
    ):
        self.s1_key = s1_key
        self.s2_key = s2_key
        self.label_key = label_key
        self.time_steps = int(time_steps)
        self.resample_size = max(0, int(resample_size))

        raw_dir = _resolve_split_dir(dataset_root, split)
        lbl_dir = _resolve_label_split_dir(label_root, split)

        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw split directory not found: {raw_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Label split directory not found: {lbl_dir}")

        raw_files = sorted(
            [
                p
                for ext in ("sample_*.npz", "sample_*.npy", "*_image.npz", "*_image.npy")
                for p in raw_dir.glob(ext)
                if p.is_file()
            ]
        )
        lbl_files = sorted(
            [
                p
                for ext in (
                    "ParcelIDs_*_labels.npz",
                    "ParcelIDs_*_labels.npy",
                    "ParcelIDs_*.npy",
                    "*_label.npz",
                    "*_label.npy",
                )
                for p in lbl_dir.glob(ext)
                if p.is_file()
            ]
        )
        if not raw_files:
            raw_files = sorted(
                [
                    p
                    for ext in ("*.npz", "*.npy")
                    for p in raw_dir.glob(ext)
                    if p.is_file() and not p.name.endswith("_label.npy") and not p.name.endswith("_label.npz")
                ]
            )
        if not lbl_files:
            # Fallback for custom label naming, but never use raw sample_*.npz as labels.
            lbl_files = sorted(
                [
                    p
                    for ext in ("*.npz", "*.npy")
                    for p in lbl_dir.glob(ext)
                    if p.is_file() and not p.name.startswith("sample_")
                ]
            )

        if not raw_files:
            raise FileNotFoundError(
                f"No raw files found in {raw_dir}. Expected patterns like sample_*.npz or *_image.npy."
            )
        if not lbl_files:
            raise FileNotFoundError(f"No label files found in {lbl_dir}")

        raw_map = _build_id_file_map(raw_files)
        lbl_map = _build_id_file_map(lbl_files)
        common_ids = sorted(set(raw_map.keys()) & set(lbl_map.keys()), key=_sort_patch_id)

        if not common_ids:
            raise RuntimeError(
                f"No matched patch IDs between raw ({raw_dir}) and labels ({lbl_dir})."
            )

        self.pairs: List[Tuple[Path, Path, str]] = [(raw_map[i], lbl_map[i], i) for i in common_ids]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_file, lbl_file, _patch_id = self.pairs[idx]

        s1_key = (self.s1_key or "").strip()
        s2_key = (self.s2_key or "").strip()
        if not s2_key:
            raise ValueError("s2_key is empty; at least one source key is required for raw training.")

        raw_data = np.load(raw_file, allow_pickle=True)
        raw_container: Optional[Dict[str, np.ndarray]] = None
        raw_array: Optional[np.ndarray] = None

        if isinstance(raw_data, np.lib.npyio.NpzFile):
            raw_container = {k: raw_data[k] for k in raw_data.files}
            raw_data.close()
        elif isinstance(raw_data, np.ndarray) and raw_data.dtype == object and raw_data.shape == ():
            obj = raw_data.item()
            if not isinstance(obj, dict):
                raise TypeError(
                    f"Unsupported object npy payload in {raw_file}: expected dict, got {type(obj)}"
                )
            raw_container = obj
        elif isinstance(raw_data, np.ndarray):
            # Support raw tensors saved directly as numpy arrays.
            raw_array = raw_data
        else:
            raise TypeError(
                f"Unsupported raw file format in {raw_file}: expected npz, dict-like npy, or ndarray npy; got {type(raw_data)}"
            )

        if raw_container is not None:
            missing = []
            if s1_key and s1_key not in raw_container:
                missing.append(s1_key)
            if s2_key not in raw_container:
                missing.append(s2_key)
            if missing:
                available = list(raw_container.keys())
                raise KeyError(
                    f"Missing source key(s) in {raw_file}. Expected {missing}. Available keys: {available}"
                )

            s1 = _to_thwc(raw_container[s1_key], key=s1_key, path=raw_file) if s1_key else None
            s2 = _to_thwc(raw_container[s2_key], key=s2_key, path=raw_file)
        else:
            # If the raw file is a plain ndarray, interpret it as the S2 source.
            if raw_array is None:
                raise RuntimeError("Internal error: raw_container and raw_array are both None")

            if raw_array.ndim == 4:
                s2 = _to_thwc(raw_array, key="raw", path=raw_file)
            elif raw_array.ndim == 3:
                # Best-effort: (T,H,W) or (H,W,C) -> add a singleton channel/time dim.
                if raw_array.shape[0] == self.time_steps:
                    s2 = raw_array[:, :, :, None]
                else:
                    s2 = raw_array[None, :, :, :]
                s2 = _to_thwc(s2, key="raw", path=raw_file)
            else:
                raise ValueError(
                    f"Unsupported ndarray raw shape in {raw_file}: expected 3D/4D, got {raw_array.shape}"
                )

            s1 = None

        if s1 is not None:
            s1 = _select_and_pad_time(_normalize_source(s1), self.time_steps)
        s2 = _select_and_pad_time(_normalize_source(s2), self.time_steps)

        x_s2 = _flatten_time_channels(s2)
        if s1 is not None:
            x_s1 = _flatten_time_channels(s1)
            x_chw = np.concatenate([x_s1, x_s2], axis=0).astype(np.float32)
        else:
            x_chw = x_s2.astype(np.float32)

        if lbl_file.suffix.lower() == ".npz":
            with np.load(lbl_file, allow_pickle=True) as lbl_npz:
                if self.label_key in lbl_npz:
                    y = lbl_npz[self.label_key]
                elif "labels" in lbl_npz:
                    y = lbl_npz["labels"]
                else:
                    available = list(lbl_npz.keys())
                    if len(available) == 1:
                        y = lbl_npz[available[0]]
                    else:
                        raise KeyError(
                            f"Label key '{self.label_key}' not found in {lbl_file}. "
                            f"Available keys: {available}"
                        )
        else:
            y_raw = np.load(lbl_file, allow_pickle=True)
            if isinstance(y_raw, np.ndarray) and y_raw.dtype == object and y_raw.shape == ():
                y_obj = y_raw.item()
                if isinstance(y_obj, dict):
                    if self.label_key in y_obj:
                        y = y_obj[self.label_key]
                    elif "labels" in y_obj:
                        y = y_obj["labels"]
                    else:
                        available = list(y_obj.keys())
                        if len(available) == 1:
                            y = y_obj[available[0]]
                        else:
                            raise KeyError(
                                f"Label key '{self.label_key}' not found in {lbl_file}. "
                                f"Available keys: {available}"
                            )
                else:
                    y = y_obj
            else:
                y = y_raw

        y = np.asarray(y).squeeze().astype(np.int64)
        if y.ndim != 2:
            raise ValueError(f"Expected 2D label in {lbl_file}, got shape {y.shape}")

        # Zero padding to 512x512
        target_h, target_w = 512, 512
        c, h, w = x_chw.shape
        if h < target_h or w < target_w:
            pad_h = target_h - h
            pad_w = target_w - w
            pad_before_h = pad_h // 2
            pad_after_h = pad_h - pad_before_h
            pad_before_w = pad_w // 2
            pad_after_w = pad_w - pad_before_w
            x_chw = np.pad(
                x_chw,
                ((0, 0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w)),
                mode="constant",
            )
            y = np.pad(
                y,
                ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w)),
                mode="constant",
                constant_values=255,  # ignore_index
            )
        elif h > target_h or w > target_w:
            # If larger, center crop
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            x_chw = x_chw[:, start_h:start_h+target_h, start_w:start_w+target_w]
            y = y[start_h:start_h+target_h, start_w:start_w+target_w]

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

    if args.time_steps <= 0:
        raise ValueError("--time_steps must be > 0 for raw training to keep fixed channel size.")

    dataset_root = Path(args.dataset_root)
    label_root = Path(args.label_root)

    if args.use_test_as_val == 1:
        effective_val_split = "test"
        print("use_test_as_val=1: validation split switched to 'test'.")
    else:
        effective_val_split = args.val_split

    train_base_ds = PastisRawSegmentationDataset(
        dataset_root=dataset_root,
        label_root=label_root,
        split=args.train_split,
        s1_key=args.s1_key,
        s2_key=args.s2_key,
        label_key=args.label_key,
        time_steps=args.time_steps,
        resample_size=args.resample_size,
    )

    val_base_ds = PastisRawSegmentationDataset(
        dataset_root=dataset_root,
        label_root=label_root,
        split=effective_val_split,
        s1_key=args.s1_key,
        s2_key=args.s2_key,
        label_key=args.label_key,
        time_steps=args.time_steps,
        resample_size=args.resample_size,
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
    elif args.model_variant == "lightweight":
        model = UNetLightweight(
            in_channels=c,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
            depth=args.unet_depth,
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

    focus_classes = _parse_focus_classes(
        focus_spec=args.focal_focus_classes,
        num_classes=args.num_classes,
        background_index=args.background_index,
        ignore_index=args.ignore_index,
    )
    print(f"Focal focus classes: {focus_classes}")

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
    best_val_f1 = 0.0
    best_val_mf1 = 0.0
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
    last_val_f1 = float("nan")
    last_val_mf1 = float("nan")
    last_val_pred_bg_ratio = float("nan")

    metric_class_indices = _metric_class_indices(
        num_classes=args.num_classes,
        background_index=args.background_index,
        ignore_index=args.ignore_index,
        ignore_background=bool(args.ignore_background_in_metrics),
    )
    print(f"Metric classes used for mF1/weighted_F1: {metric_class_indices}")

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
        val_running_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
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
                val_running_confusion += compute_confusion_matrix(
                    logits,
                    y,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                )

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
        val_precision, val_recall, val_f1, val_macro_precision, val_macro_recall, val_mf1, val_weighted_f1 = compute_f1_from_confusion_matrix(
            val_running_confusion,
            background_index=args.background_index,
            ignore_background=bool(args.ignore_background_in_metrics),
            class_indices=metric_class_indices,
        )
        val_mf1_from_f1 = float(np.mean(val_f1[metric_class_indices])) if metric_class_indices else 0.0
        class_support = val_running_confusion.sum(axis=1)
        support = class_support[metric_class_indices]
        support_sum = float(np.sum(support))
        val_weighted_f1_from_f1 = (
            float(np.sum(val_f1[metric_class_indices] * support) / support_sum)
            if support_sum > 0.0
            else 0.0
        )
        if not np.isclose(val_mf1, val_mf1_from_f1, atol=1e-8):
            raise RuntimeError(
                f"mF1 mismatch: computed={val_mf1:.10f}, from_per_class={val_mf1_from_f1:.10f}"
            )
        if not np.isclose(val_weighted_f1, val_weighted_f1_from_f1, atol=1e-8):
            raise RuntimeError(
                f"weighted F1 mismatch: computed={val_weighted_f1:.10f}, from_per_class={val_weighted_f1_from_f1:.10f}"
            )

        last_train_loss = float(train_loss)
        last_train_acc = float(train_acc)
        last_train_miou = float(train_miou)
        last_train_pred_bg_ratio = float(train_pred_bg_ratio)
        last_val_loss = float(val_loss)
        last_val_acc = float(val_acc)
        last_val_miou = float(val_miou)
        last_val_f1 = float(val_mf1)
        last_val_mf1 = float(val_mf1)
        last_val_pred_bg_ratio = float(val_pred_bg_ratio)

        improved = (val_miou > best_val_miou) or (np.isclose(val_miou, best_val_miou) and val_loss < best_val_loss)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_miou = val_miou
            best_val_f1 = val_mf1
            best_val_mf1 = val_mf1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
            f"train_pred_bg: {train_pred_bg_ratio:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mIoU: {val_miou:.4f}, val_mF1: {val_mf1:.4f}, "
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

    ckpt_latest = out_dir / "unet_raw_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "in_channels": c}, ckpt_latest)
    print(f"Saved latest checkpoint to {ckpt_latest}")

    if best_state is not None:
        ckpt_best = out_dir / "unet_raw_best.pt"
        torch.save({"model_state_dict": best_state, "in_channels": c}, ckpt_best)
        print(
            f"Saved best checkpoint to {ckpt_best} "
            f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, val_mIoU={best_val_miou:.4f})"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        final_val_miou_sum = 0.0
        final_val_miou_count = 0
        final_val_pred_bg = 0
        final_val_pred_total = 0
        final_val_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                logits, y = align_logits_and_target_spatial(logits, y)
                loss_focal = criterion_focal(logits, y)
                loss_dice = criterion_dice(logits, y)
                loss = loss_focal + args.dice_weight * loss_dice
                final_val_loss += loss.item() * x.size(0)

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
                final_val_correct += int(round(acc * total))
                final_val_total += int(total)
                pred = logits.argmax(dim=1)
                final_val_pred_bg += int(((pred == args.background_index) & mask).sum().item())
                final_val_pred_total += int(total)
                final_val_miou_sum += miou
                final_val_miou_count += 1
                final_val_confusion += compute_confusion_matrix(
                    logits,
                    y,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                )

        final_val_loss = final_val_loss / max(1, len(val_ds))
        final_val_acc = final_val_correct / max(1, final_val_total) if final_val_total > 0 else 0.0
        final_val_miou = final_val_miou_sum / max(1, final_val_miou_count)
        final_val_pred_bg_ratio = final_val_pred_bg / max(1, final_val_pred_total)
        final_precision, final_recall, final_f1, final_macro_precision, final_macro_recall, final_mf1, final_weighted_f1 = compute_f1_from_confusion_matrix(
            final_val_confusion,
            background_index=args.background_index,
            ignore_background=bool(args.ignore_background_in_metrics),
            class_indices=metric_class_indices,
        )
        final_mf1_from_f1 = float(np.mean(final_f1[metric_class_indices])) if metric_class_indices else 0.0
        final_class_support = final_val_confusion.sum(axis=1)
        final_support = final_class_support[metric_class_indices]
        final_support_sum = float(np.sum(final_support))
        final_weighted_f1_from_f1 = (
            float(np.sum(final_f1[metric_class_indices] * final_support) / final_support_sum)
            if final_support_sum > 0.0
            else 0.0
        )
        if not np.isclose(final_mf1, final_mf1_from_f1, atol=1e-8):
            raise RuntimeError(
                f"Final mF1 mismatch: computed={final_mf1:.10f}, from_per_class={final_mf1_from_f1:.10f}"
            )
        if not np.isclose(final_weighted_f1, final_weighted_f1_from_f1, atol=1e-8):
            raise RuntimeError(
                f"Final weighted F1 mismatch: computed={final_weighted_f1:.10f}, from_per_class={final_weighted_f1_from_f1:.10f}"
            )
        final_val_f1 = float(final_mf1)
        final_val_mf1 = float(final_mf1)

        per_class_iou = evaluate_per_class_iou(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            background_index=args.background_index,
            ignore_background=args.ignore_background_in_metrics,
        )
        dump_per_class_iou_report(
            per_class_iou,
            output_dir=out_dir,
            prefix="best_val",
            per_class_f1=final_f1,
            weighted_f1=final_weighted_f1,
        )
        try:
            from seg.train_unet_from_embeddings import save_confusion_matrix_visualization
        except ModuleNotFoundError:
            from train_unet_from_embeddings import save_confusion_matrix_visualization

        save_confusion_matrix_visualization(
            final_val_confusion,
            output_dir=out_dir,
            prefix="best_val",
            class_names=[str(i) for i in range(args.num_classes)],
            normalize=True,
        )
        print(
            f"Best checkpoint validation: loss={final_val_loss:.4f}, acc={final_val_acc:.4f}, "
            f"mIoU={final_val_miou:.4f}, mF1={final_val_mf1:.4f}, pred_bg={final_val_pred_bg_ratio:.4f}"
        )

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
        "script": "train_unet_from_mts12_raw.py",
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
            "best_val_f1": float(best_val_f1),
            "best_val_mf1": float(best_val_mf1),
        },
        "last_metrics": {
            "train_loss": last_train_loss,
            "train_acc": last_train_acc,
            "train_miou": last_train_miou,
            "train_pred_bg_ratio": last_train_pred_bg_ratio,
            "val_loss": last_val_loss,
            "val_acc": last_val_acc,
            "val_miou": last_val_miou,
            "val_f1": last_val_f1,
            "val_mf1": last_val_mf1,
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
            "dataset_format": args.dataset_format,
            "dataset_root": str(dataset_root),
            "label_root": str(label_root),
            "train_split": args.train_split,
            "val_split": effective_val_split,
            "time_steps": int(args.time_steps),
            "s1_key": args.s1_key,
            "s2_key": args.s2_key,
            "label_key": args.label_key,
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
            "Train and validate U-Net segmentation on raw Pastis-R time-series data "
            "(e.g. sentinel1/sentinel2 from sample_*.npz)."
        )
    )

    p.add_argument("--dataset_format", type=str, default="pastis_raw", choices=["pastis_raw", "mts12_raw"])
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--label_root", type=str, required=True)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--use_test_as_val", type=int, default=0, choices=[0, 1])

    p.add_argument("--s1_key", type=str, default="sentinel1")
    p.add_argument("--s2_key", type=str, default="sentinel2")
    p.add_argument("--label_key", type=str, default="labels")
    p.add_argument("--time_steps", type=int, default=12)
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
    p.add_argument(
        "--model_variant",
        type=str,
        default="basic",
        choices=["basic", "resse", "lightweight", "aspp"],
        help="Model variant: basic VGG16-style UNet, resse UNet, lightweight UNet, or aspp DeepLabLite",
    )
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--norm_type", type=str, default="instance", choices=["instance", "group", "batch"])
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
