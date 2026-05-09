import argparse
import contextlib
import copy
import csv
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    import yaml
except ImportError:
    yaml = None

try:
    from seg.train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        EmbeddingSegmentationDataset,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        background_ratio_loss,
        compute_confusion_matrix,
        compute_f1_from_confusion_matrix,
        foreground_aux_loss,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetLightweight,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        dump_per_class_iou_report,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        save_confusion_matrix_visualization,
        save_val_visualizations,
    )
    from seg.train_unet_from_mts12_raw import PastisRawSegmentationDataset
except ModuleNotFoundError:
    from train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        EmbeddingSegmentationDataset,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        background_ratio_loss,
        compute_confusion_matrix,
        compute_f1_from_confusion_matrix,
        foreground_aux_loss,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetLightweight,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        dump_per_class_iou_report,
        estimate_feature_channel_stats,
        estimate_label_valid_ratio,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        save_confusion_matrix_visualization,
        save_val_visualizations,
    )
    from train_unet_from_mts12_raw import PastisRawSegmentationDataset


def _load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML is not installed. Please install pyyaml or use JSON config.")
            cfg = yaml.safe_load(f)
        elif suffix == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/object.")
    return cfg


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def _as_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _as_str(cfg: Dict[str, Any], key: str, default: str) -> str:
    return str(cfg.get(key, default))


def _as_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, np.integer)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(raw)


def _parse_float_list(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    s = str(raw).strip()
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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

    include = set()
    for tok in [x.strip().lower() for x in raw.split(",") if x.strip()]:
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


def _select_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_model_meta_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    keys = set(state.keys())

    if any(k.startswith("aspp.") for k in keys):
        variant = "aspp"
        depth = 4
        base_ch = int(state["stem.conv.weight"].shape[0])
    elif any(k.startswith("enc_blocks.0.block.0.depthwise") for k in keys):
        variant = "lightweight"
        depth = 5 if any(k.startswith("enc_blocks.4.") for k in keys) else 4
        base_ch = int(state["enc_blocks.0.block.0.depthwise.weight"].shape[0])
    elif any(k.startswith("enc1.block1.conv1") for k in keys):
        variant = "resse"
        depth = 5 if any(k.startswith("enc5.") for k in keys) else 4
        base_ch = int(state["enc1.block1.conv1.weight"].shape[0])
    else:
        variant = "basic"
        depth = 5 if any(k.startswith("enc5.") for k in keys) else 4
        base_ch = int(state["enc1.net.0.weight"].shape[0])

    num_classes = int(state["out_conv.weight"].shape[0])
    return {
        "variant": variant,
        "depth": depth,
        "base_channels": base_ch,
        "num_classes": num_classes,
    }


def _build_model(
    in_channels: int,
    num_classes: int,
    model_cfg: Dict[str, Any],
) -> nn.Module:
    variant = _as_str(model_cfg, "variant", "basic")
    base_channels = _as_int(model_cfg, "base_channels", 32)
    depth = _as_int(model_cfg, "depth", 4)
    norm_type = _as_str(model_cfg, "norm", "group")
    dropout_rate = _as_float(model_cfg, "dropout", 0.1)

    if variant == "aspp":
        return DeepLabLite(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
        )
    if variant == "resse":
        return UNetResSE(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
            depth=depth,
            dropout=dropout_rate,
        )
    if variant == "lightweight":
        return UNetLightweight(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
            depth=depth,
        )
    if depth == 5:
        return UNetDeep(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_channels,
            norm=norm_type,
        )
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_channels,
        norm=norm_type,
    )


def _build_embedding_dataset(cfg: Dict[str, Any], split: str) -> EmbeddingSegmentationDataset:
    emb_cfg = cfg["embedding"]
    key = f"{split}_embeddings_path"
    lbl_key = f"{split}_labels_path"
    if key not in emb_cfg or lbl_key not in emb_cfg:
        raise KeyError(f"Missing embedding config keys: {key}, {lbl_key}")

    emb_path = Path(str(emb_cfg[key]))
    lbl_path = Path(str(emb_cfg[lbl_key]))
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {emb_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Labels path not found: {lbl_path}")

    per_patch_labels = bool(_as_bool(emb_cfg, "per_patch_labels", False) or lbl_path.is_dir())
    return EmbeddingSegmentationDataset(
        embeddings_path=emb_path,
        labels_path=lbl_path,
        per_patch_labels=per_patch_labels,
        embedding_key=_as_str(emb_cfg, "embedding_key", "auto"),
        resample_size=_as_int(cfg.get("data", {}), "resample_size", 0),
    )


def _build_raw_dataset(cfg: Dict[str, Any], split: str) -> PastisRawSegmentationDataset:
    raw_cfg = cfg["raw"]
    dataset_root = Path(_as_str(raw_cfg, "dataset_root", ""))
    label_root = Path(_as_str(raw_cfg, "label_root", ""))
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label_root not found: {label_root}")

    split_name = _as_str(raw_cfg, f"{split}_split", split)
    return PastisRawSegmentationDataset(
        dataset_root=dataset_root,
        label_root=label_root,
        split=split_name,
        s1_key=_as_str(raw_cfg, "s1_key", "sentinel1"),
        s2_key=_as_str(raw_cfg, "s2_key", "sentinel2"),
        label_key=_as_str(raw_cfg, "label_key", "labels"),
        time_steps=_as_int(raw_cfg, "time_steps", 12),
        resample_size=_as_int(cfg.get("data", {}), "resample_size", 0),
    )


def _build_train_val_datasets(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset, Optional[np.ndarray]]:
    input_cfg = cfg["input"]
    mode = _as_str(input_cfg, "mode", "embedding")
    train_cfg = cfg.get("train", {})

    if mode == "embedding":
        train_base = _build_embedding_dataset(cfg, "train")
        emb_cfg = cfg["embedding"]
        has_explicit_val = bool(emb_cfg.get("val_embeddings_path")) and bool(emb_cfg.get("val_labels_path"))
        if has_explicit_val:
            val_base = _build_embedding_dataset(cfg, "val")
            train_indices = np.arange(len(train_base), dtype=int)
        else:
            if len(train_base) < 2:
                raise ValueError("Embedding train dataset has less than 2 samples; cannot split train/val.")
            val_fraction = _as_float(train_cfg, "val_fraction", 0.2)
            n_total = len(train_base)
            n_val = max(1, int(n_total * val_fraction))
            n_train = n_total - n_val
            if n_train <= 0:
                raise ValueError("val_fraction leaves no training samples.")
            rng = np.random.default_rng(_as_int(train_cfg, "seed", 42))
            indices = rng.permutation(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            train_base = Subset(train_base, train_indices.tolist())
            val_base = Subset(_build_embedding_dataset(cfg, "train"), val_indices.tolist())
    elif mode == "raw":
        train_base = _build_raw_dataset(cfg, "train")
        val_base = _build_raw_dataset(cfg, "val")
        train_indices = np.arange(len(train_base), dtype=int)
    else:
        raise ValueError(f"Unsupported input.mode: {mode}")

    base_train_for_sampling = train_base
    data_cfg = cfg.get("data", {})

    if _as_bool(data_cfg, "normalize_features", True) and len(train_base) > 0:
        feat_mean, feat_std = estimate_feature_channel_stats(
            train_base,
            max_samples=_as_int(data_cfg, "feature_stats_max_samples", 0),
        )
        train_base = FeatureNormalizeDataset(train_base, feat_mean, feat_std)
        val_base = FeatureNormalizeDataset(val_base, feat_mean, feat_std)

    train_ds: Dataset = train_base
    if _as_bool(data_cfg, "enable_augmentation", True) and len(train_base) > 0:
        train_ds = AugmentedTrainDataset(
            base_dataset=train_base,
            repeat_factor=_as_int(data_cfg, "train_repeat_factor", 1),
            flip_prob=_as_float(data_cfg, "aug_flip_prob", 0.5),
            rot90_prob=_as_float(data_cfg, "aug_rot90_prob", 0.5),
            noise_std=_as_float(data_cfg, "aug_noise_std", 0.01),
            gain_std=_as_float(data_cfg, "aug_gain_std", 0.05),
            crop_size=_as_int(data_cfg, "train_crop_size", 0),
            background_index=_as_int(cfg.get("labels", {}), "background_index", 0),
            ignore_index=_as_int(cfg.get("labels", {}), "ignore_index", 255),
        )

    return train_ds, val_base, base_train_for_sampling, train_indices


def _build_loader(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler] = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    return DataLoader(
        **loader_kwargs,
    )


def _compute_losses(
    logits: torch.Tensor,
    y: torch.Tensor,
    criterion_focal: nn.Module,
    criterion_dice: nn.Module,
    criterion_ce: Optional[nn.Module],
    fg_aux_weight: float,
    bg_ratio_weight: float,
    background_index: int,
    ignore_index: int,
    dice_weight: float,
    ce_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_focal = criterion_focal(logits, y)
    loss_dice = criterion_dice(logits, y)
    loss_fg_aux = foreground_aux_loss(
        logits,
        y,
        background_index=background_index,
        ignore_index=ignore_index,
    )
    loss_bg_ratio = background_ratio_loss(
        logits,
        y,
        background_index=background_index,
        ignore_index=ignore_index,
    )
    loss = loss_focal + dice_weight * loss_dice
    if fg_aux_weight > 0.0:
        loss = loss + fg_aux_weight * loss_fg_aux
    if bg_ratio_weight > 0.0:
        loss = loss + bg_ratio_weight * loss_bg_ratio

    parts = {
        "focal": float(loss_focal.item()),
        "dice": float(loss_dice.item()),
        "fg_aux": float(loss_fg_aux.item()),
        "bg_ratio": float(loss_bg_ratio.item()),
    }

    if criterion_ce is not None and ce_weight > 0.0:
        loss_ce = criterion_ce(logits, y)
        loss = loss + ce_weight * loss_ce
        parts["ce"] = float(loss_ce.item())

    parts["total"] = float(loss.item())
    return loss, parts


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    background_index: int,
    ignore_background_in_metrics: bool,
    criterion_focal: nn.Module,
    criterion_dice: nn.Module,
    criterion_ce: Optional[nn.Module],
    fg_aux_weight: float,
    bg_ratio_weight: float,
    dice_weight: float,
    ce_weight: float,
    optimizer: Optional[torch.optim.Optimizer],
    amp_enabled: bool,
    grad_scaler: Optional[torch.amp.GradScaler],
    use_channels_last: bool,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_miou_sum = 0.0
    running_miou_count = 0
    running_pred_bg = 0
    running_pred_total = 0
    running_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = model(x)
                logits, y = align_logits_and_target_spatial(logits, y)
                loss, _ = _compute_losses(
                    logits,
                    y,
                    criterion_focal,
                    criterion_dice,
                    criterion_ce,
                    fg_aux_weight,
                    bg_ratio_weight,
                    background_index,
                    ignore_index,
                    dice_weight,
                    ce_weight,
                )

            if is_train:
                if grad_scaler is not None and amp_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        running_loss += float(loss.item()) * x.size(0)

        with torch.no_grad():
            acc, miou = compute_segmentation_metrics(
                logits,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
                background_index=background_index,
                ignore_background=ignore_background_in_metrics,
            )
            mask = y != ignore_index
            total = int(mask.sum().item())
            running_correct += int(round(acc * total))
            running_total += total
            pred = logits.argmax(dim=1)
            running_pred_bg += int(((pred == background_index) & mask).sum().item())
            running_pred_total += total
            running_miou_sum += float(miou)
            running_miou_count += 1
            running_confusion += compute_confusion_matrix(
                logits,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

    avg_loss = running_loss / max(1, len(loader.dataset))
    avg_acc = running_correct / max(1, running_total) if running_total > 0 else 0.0
    avg_miou = running_miou_sum / max(1, running_miou_count)
    pred_bg_ratio = running_pred_bg / max(1, running_pred_total)
    precision, recall, f1, macro_precision, macro_recall, macro_f1, weighted_f1 = compute_f1_from_confusion_matrix(
        running_confusion,
        background_index=background_index,
        ignore_background=ignore_background_in_metrics,
    )

    return {
        "loss": float(avg_loss),
        "acc": float(avg_acc),
        "miou": float(avg_miou),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "mf1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": running_confusion,
        "pred_bg_ratio": float(pred_bg_ratio),
    }


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    torch.save(payload, path)


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        return obj
    return {"model_state_dict": obj}


def _save_history_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_miou",
        "train_f1",
        "train_mf1",
        "train_pred_bg_ratio",
        "val_loss",
        "val_acc",
        "val_miou",
        "val_f1",
        "val_mf1",
        "val_pred_bg_ratio",
        "lr",
        "is_best",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _default_train_config() -> Dict[str, Any]:
    return {
        "input": {"mode": "embedding"},
        "labels": {
            "num_classes": 9,
            "ignore_index": 255,
            "background_index": 0,
            "ignore_background_in_metrics": True,
            "ignore_background_in_dice": True,
        },
        "model": {
            "variant": "basic",
            "depth": 4,
            "base_channels": 32,
            "norm": "group",
            "dropout": 0.1,
        },
        "train": {
            "epochs": 50,
            "batch_size": 4,
            "num_workers": 4,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "lr": 5e-5,
            "weight_decay": 1e-4,
            "seed": 42,
            "val_fraction": 0.2,
            "val_interval": 1,
            "early_stop_patience": 0,
            "scheduler_monitor": "val_miou",
            "lr_patience": 5,
            "lr_decay_factor": 0.5,
            "min_lr": 1e-6,
            "dice_weight": 0.4,
            "ce_weight": 0.0,
            "focal_gamma": 2.0,
            "focal_class_weights": "",
            "focal_focus_classes": "all_fg",
            "save_val_samples": 6,
        },
        "data": {
            "resample_size": 0,
            "normalize_features": True,
            "feature_stats_max_samples": 0,
            "enable_augmentation": True,
            "train_repeat_factor": 1,
            "aug_flip_prob": 0.5,
            "aug_rot90_prob": 0.5,
            "aug_noise_std": 0.01,
            "aug_gain_std": 0.05,
            "train_crop_size": 0,
            "fg_crop_prob": 0.5,
            "use_fg_sampler": False,
            "fg_sampler_power": 1.5,
            "fg_sampler_min_weight": 0.2,
        },
        "runtime": {
            "device": "",
            "output_dir": "",
            "resume": False,
            "resume_checkpoint": "",
            "resume_strict": True,
            "amp": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
            "channels_last": True,
            "matmul_precision": "high",
        },
        "search": {
            "enabled": False,
            "grid": {
                "train.lr": [],
                "model.base_channels": [],
                "train.dice_weight": [],
            },
        },
    }


def _materialize_run_config(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    merged = _merge_dict(_default_train_config(), cfg)

    runtime = merged.setdefault("runtime", {})
    if args.device:
        runtime["device"] = args.device
    if output_dir_override is not None:
        runtime["output_dir"] = output_dir_override
    elif args.output_dir:
        runtime["output_dir"] = args.output_dir

    runtime["resume"] = bool(getattr(args, "resume", False))
    resume_checkpoint = getattr(args, "resume_checkpoint", "")
    if resume_checkpoint:
        runtime["resume_checkpoint"] = resume_checkpoint

    if not runtime.get("output_dir"):
        raise ValueError("runtime.output_dir is required (or use --output_dir).")

    return merged


def _train_once(cfg: Dict[str, Any]) -> Dict[str, Any]:
    labels_cfg = cfg["labels"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    seed = _as_int(train_cfg, "seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _select_device(_as_str(runtime_cfg, "device", ""))
    amp_enabled = bool(device.type == "cuda" and _as_bool(runtime_cfg, "amp", True))
    use_channels_last = bool(device.type == "cuda" and _as_bool(runtime_cfg, "channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = _as_bool(runtime_cfg, "cudnn_benchmark", True)
        allow_tf32 = _as_bool(runtime_cfg, "allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        matmul_precision = _as_str(runtime_cfg, "matmul_precision", "high").strip().lower()
        if matmul_precision in ("high", "medium", "highest"):
            torch.set_float32_matmul_precision(matmul_precision)

    output_dir = Path(_as_str(runtime_cfg, "output_dir", ""))
    _ensure_dir(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    _ensure_dir(checkpoints_dir)

    train_ds, val_ds, base_train_for_sampling, _ = _build_train_val_datasets(cfg)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train or val dataset is empty.")

    sample_feats, sample_labels = train_ds[0]
    in_channels = int(sample_feats.shape[0])

    num_classes = _as_int(labels_cfg, "num_classes", 9)
    ignore_index = _as_int(labels_cfg, "ignore_index", 255)
    background_index = _as_int(labels_cfg, "background_index", 0)
    ignore_bg_metrics = _as_bool(labels_cfg, "ignore_background_in_metrics", True)
    ignore_bg_dice = _as_bool(labels_cfg, "ignore_background_in_dice", True)

    valid_ratio, y_min, y_max = estimate_label_valid_ratio(
        base_train_for_sampling,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_samples=16,
    )
    if valid_ratio < 0.90:
        raise ValueError(
            "Label sanity check failed: valid ratio too low. "
            f"valid_ratio={valid_ratio:.4f}, min={y_min}, max={y_max}, "
            f"num_classes={num_classes}, ignore_index={ignore_index}"
        )

    sampler = None
    if _as_bool(data_cfg, "use_fg_sampler", False):
        fg_ratios = estimate_sample_foreground_ratios(
            base_train_for_sampling,
            background_index=background_index,
            ignore_index=ignore_index,
        )
        weights = _as_float(data_cfg, "fg_sampler_min_weight", 0.2) + np.power(
            np.clip(fg_ratios, 1e-6, None),
            _as_float(data_cfg, "fg_sampler_power", 1.5),
        )
        if len(train_ds) != len(base_train_for_sampling):
            rep = int(np.ceil(len(train_ds) / len(base_train_for_sampling)))
            weights = np.tile(weights, rep)[: len(train_ds)]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )

    batch_size = _as_int(train_cfg, "batch_size", 4)
    num_workers = _as_int(train_cfg, "num_workers", 4)
    persistent_workers = _as_bool(train_cfg, "persistent_workers", num_workers > 0)
    prefetch_factor = _as_int(train_cfg, "prefetch_factor", 2)
    pin_memory = _as_bool(runtime_cfg, "pin_memory", True)

    train_loader = _build_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=True,
        sampler=sampler,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    val_loader = _build_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    model = _build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        model_cfg=cfg["model"],
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    focal_focus_classes = _parse_focus_classes(
        _as_str(train_cfg, "focal_focus_classes", "all_fg"),
        num_classes=num_classes,
        background_index=background_index,
        ignore_index=ignore_index,
    )

    criterion_focal = FocalLossWithSampling(
        num_classes=num_classes,
        gamma=_as_float(train_cfg, "focal_gamma", 2.0),
        ignore_index=ignore_index,
        focus_classes=focal_focus_classes,
        class_weights=_parse_float_list(train_cfg.get("focal_class_weights")),
    )
    criterion_dice = SoftDiceLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background=ignore_bg_dice,
    )

    ce_weight = _as_float(train_cfg, "ce_weight", 0.0)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if ce_weight > 0.0 else None
    fg_aux_weight = _as_float(train_cfg, "fg_aux_weight", 0.2)
    bg_ratio_weight = _as_float(train_cfg, "bg_ratio_weight", 0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_as_float(train_cfg, "lr", 5e-5),
        weight_decay=_as_float(train_cfg, "weight_decay", 1e-4),
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    scheduler_mode = "max" if _as_str(train_cfg, "scheduler_monitor", "val_miou") == "val_miou" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=_as_float(train_cfg, "lr_decay_factor", 0.5),
        patience=_as_int(train_cfg, "lr_patience", 5),
        min_lr=_as_float(train_cfg, "min_lr", 1e-6),
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_miou = 0.0
    best_val_f1 = 0.0
    best_val_mf1 = 0.0
    best_state = None
    history_rows: List[Dict[str, Any]] = []

    do_resume = _as_bool(runtime_cfg, "resume", False)

    final_val_stats: Optional[Dict[str, Any]] = None
    if do_resume:
        resume_path = _as_str(runtime_cfg, "resume_checkpoint", "")
        if not resume_path:
            resume_path = str(checkpoints_dir / "latest.pt")
        resume_ckpt = _load_checkpoint(Path(resume_path), device=device)

        model_state = resume_ckpt.get("model_state_dict", resume_ckpt)
        strict = _as_bool(runtime_cfg, "resume_strict", True)
        model.load_state_dict(model_state, strict=strict)

        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_val_loss = float(resume_ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(resume_ckpt.get("best_val_acc", best_val_acc))
        best_val_miou = float(resume_ckpt.get("best_val_miou", best_val_miou))
        best_val_f1 = float(resume_ckpt.get("best_val_f1", best_val_f1))
        best_val_mf1 = float(resume_ckpt.get("best_val_mf1", best_val_mf1))

    epochs = _as_int(train_cfg, "epochs", 50)
    val_interval = max(1, _as_int(train_cfg, "val_interval", 1))
    validate_first_epoch = _as_bool(train_cfg, "validate_first_epoch", True)
    early_stop_patience = _as_int(train_cfg, "early_stop_patience", 0)
    epochs_no_improve = 0
    run_start = time.time()

    history_csv = output_dir / "metrics_history.csv"

    for epoch in range(start_epoch, epochs):
        epoch_idx = epoch + 1
        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            background_index=background_index,
            ignore_background_in_metrics=ignore_bg_metrics,
            criterion_focal=criterion_focal,
            criterion_dice=criterion_dice,
            criterion_ce=criterion_ce,
            fg_aux_weight=fg_aux_weight,
            bg_ratio_weight=bg_ratio_weight,
            dice_weight=_as_float(train_cfg, "dice_weight", 0.4),
            ce_weight=ce_weight,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            grad_scaler=grad_scaler,
            use_channels_last=use_channels_last,
        )
        run_val = (epoch_idx % val_interval == 0) or (epoch_idx == epochs)
        if validate_first_epoch and epoch_idx == 1:
            run_val = True
        val_stats: Optional[Dict[str, float]] = None
        improved = False
        if run_val:
            val_stats = _run_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
                background_index=background_index,
                ignore_background_in_metrics=ignore_bg_metrics,
                criterion_focal=criterion_focal,
                criterion_dice=criterion_dice,
                criterion_ce=criterion_ce,
                fg_aux_weight=fg_aux_weight,
                bg_ratio_weight=bg_ratio_weight,
                dice_weight=_as_float(train_cfg, "dice_weight", 0.4),
                ce_weight=ce_weight,
                optimizer=None,
                amp_enabled=amp_enabled,
                grad_scaler=grad_scaler,
                use_channels_last=use_channels_last,
            )

            monitor = _as_str(train_cfg, "scheduler_monitor", "val_miou")
            if monitor == "val_miou":
                scheduler.step(val_stats["miou"])
            else:
                scheduler.step(val_stats["loss"])

            improved = (val_stats["miou"] > best_val_miou) or (
                np.isclose(val_stats["miou"], best_val_miou) and val_stats["loss"] < best_val_loss
            )

            if improved:
                best_val_loss = val_stats["loss"]
                best_val_acc = val_stats["acc"]
                best_val_miou = val_stats["miou"]
                best_val_f1 = val_stats["f1"]
                best_val_mf1 = val_stats["mf1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0

                _save_checkpoint(
                    checkpoints_dir / "best.pt",
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        "best_val_miou": best_val_miou,
                        "best_val_f1": best_val_f1,
                        "best_val_mf1": best_val_mf1,
                        "in_channels": in_channels,
                        "num_classes": num_classes,
                        "model_cfg": cfg["model"],
                    },
                )
            else:
                epochs_no_improve += 1

        latest_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_miou": best_val_miou,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "model_cfg": cfg["model"],
        }
        _save_checkpoint(checkpoints_dir / "latest.pt", latest_payload)

        row = {
            "epoch": epoch_idx,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "train_miou": train_stats["miou"],
            "train_f1": train_stats["f1"],
            "train_mf1": train_stats["mf1"],
            "train_pred_bg_ratio": train_stats["pred_bg_ratio"],
            "val_loss": float(val_stats["loss"]) if val_stats is not None else float("nan"),
            "val_acc": float(val_stats["acc"]) if val_stats is not None else float("nan"),
            "val_miou": float(val_stats["miou"]) if val_stats is not None else float("nan"),
            "val_f1": float(val_stats["f1"]) if val_stats is not None else float("nan"),
            "val_mf1": float(val_stats["mf1"]) if val_stats is not None else float("nan"),
            "val_pred_bg_ratio": float(val_stats["pred_bg_ratio"]) if val_stats is not None else float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "is_best": int(improved),
        }
        history_rows.append(row)
        _save_history_csv(history_csv, history_rows)

        if val_stats is not None:
            print(
                f"Epoch {epoch_idx}/{epochs} | "
                f"train_loss={train_stats['loss']:.4f}, train_mIoU={train_stats['miou']:.4f}, "
                f"train_mF1={train_stats['mf1']:.4f}, "
                f"val_loss={val_stats['loss']:.4f}, val_mIoU={val_stats['miou']:.4f}, val_mF1={val_stats['mf1']:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(
                f"Epoch {epoch_idx}/{epochs} | "
                f"train_loss={train_stats['loss']:.4f}, train_mIoU={train_stats['miou']:.4f}, train_mF1={train_stats['mf1']:.4f}, "
                f"val=skipped(interval={val_interval}), lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_stats is not None and early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch_idx} (no mIoU improvement for {early_stop_patience} epoch(s))."
            )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)

    final_val_stats = _run_one_epoch(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background_in_metrics=ignore_bg_metrics,
        criterion_focal=criterion_focal,
        criterion_dice=criterion_dice,
        criterion_ce=criterion_ce,
        fg_aux_weight=fg_aux_weight,
        bg_ratio_weight=bg_ratio_weight,
        dice_weight=_as_float(train_cfg, "dice_weight", 0.4),
        ce_weight=ce_weight,
        optimizer=None,
        amp_enabled=amp_enabled,
        grad_scaler=None,
        use_channels_last=use_channels_last,
    )

    _save_checkpoint(
        checkpoints_dir / "final.pt",
        {
            "epoch": len(history_rows) - 1,
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_val_miou": best_val_miou,
            "best_val_f1": best_val_f1,
            "best_val_mf1": best_val_mf1,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "model_cfg": cfg["model"],
        },
    )

    per_class_iou = evaluate_per_class_iou(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background=ignore_bg_metrics,
    )
    dump_per_class_iou_report(
        per_class_iou,
        output_dir=output_dir,
        prefix="best_val",
        per_class_f1=final_val_stats["per_class_f1"],
        weighted_f1=final_val_stats["weighted_f1"],
    )
    save_confusion_matrix_visualization(
        final_val_stats["confusion_matrix"],
        output_dir=output_dir,
        prefix="best_val",
        class_names=[str(i) for i in range(num_classes)],
        normalize=True,
    )

    save_val_visualizations(
        model=model,
        val_ds=val_ds,
        device=device,
        output_dir=output_dir,
        num_classes=num_classes,
        ignore_index=ignore_index,
        max_images=_as_int(train_cfg, "save_val_samples", 6),
    )

    try:
        import matplotlib.pyplot as plt

        if history_rows:
            epochs_arr = [r["epoch"] for r in history_rows]
            train_loss = [r["train_loss"] for r in history_rows]
            val_loss = [r["val_loss"] for r in history_rows]
            train_miou = [r["train_miou"] for r in history_rows]
            val_miou = [r["val_miou"] for r in history_rows]

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_arr, train_loss, label="train_loss")
            plt.plot(epochs_arr, val_loss, label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_arr, train_miou, label="train_mIoU")
            plt.plot(epochs_arr, val_miou, label="val_mIoU")
            plt.xlabel("epoch")
            plt.ylabel("mIoU")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "training_curves.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"Warning: failed to save training curves: {e}")

    resolved_config_path = output_dir / "resolved_config.json"
    _write_json(resolved_config_path, cfg)

    best_hparams = {
        "lr": _as_float(train_cfg, "lr", 5e-5),
        "dice_weight": _as_float(train_cfg, "dice_weight", 0.4),
        "ce_weight": _as_float(train_cfg, "ce_weight", 0.0),
        "focal_gamma": _as_float(train_cfg, "focal_gamma", 2.0),
        "batch_size": _as_int(train_cfg, "batch_size", 4),
        "base_channels": _as_int(cfg["model"], "base_channels", 32),
        "depth": _as_int(cfg["model"], "depth", 4),
        "variant": _as_str(cfg["model"], "variant", "basic"),
    }
    _write_json(output_dir / "best_hparams.json", best_hparams)

    final_result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": float(time.time() - run_start),
        "output_dir": str(output_dir),
        "epochs_target": epochs,
        "epochs_ran": len(history_rows),
        "input_mode": _as_str(cfg["input"], "mode", "embedding"),
        "best_metrics": {
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "best_val_miou": float(best_val_miou),
            "best_val_f1": float(best_val_f1),
            "best_val_mf1": float(best_val_mf1),
            "best_val_precision": float(final_val_stats["precision"]),
            "best_val_recall": float(final_val_stats["recall"]),
        },
        "checkpoint": {
            "latest": str(checkpoints_dir / "latest.pt"),
            "best": str(checkpoints_dir / "best.pt"),
            "final": str(checkpoints_dir / "final.pt"),
        },
    }
    _write_json(output_dir / "final_result.json", final_result)
    return final_result


def _iter_grid(grid_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = []
    value_lists = []
    for key, values in grid_cfg.items():
        if not isinstance(values, list) or len(values) == 0:
            continue
        keys.append(str(key))
        value_lists.append(values)

    if not keys:
        return []

    combos = []
    for prod in itertools.product(*value_lists):
        combo = {}
        for k, v in zip(keys, prod):
            combo[k] = v
        combos.append(combo)
    return combos


def _run_train_command(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    base_cfg = _materialize_run_config(cfg, args)

    output_dir = Path(base_cfg["runtime"]["output_dir"])
    _ensure_dir(output_dir)

    search_cfg = base_cfg.get("search", {})
    if _as_bool(search_cfg, "enabled", False):
        combos = _iter_grid(search_cfg.get("grid", {}))
        if not combos:
            raise ValueError("search.enabled=true but search.grid has no non-empty candidate list.")

        print(f"Search enabled, total trials: {len(combos)}")
        all_results: List[Dict[str, Any]] = []
        best_trial = None
        best_miou = -1.0

        for idx, combo in enumerate(combos, start=1):
            trial_cfg = copy.deepcopy(base_cfg)
            trial_cfg["runtime"]["resume"] = False
            trial_cfg["runtime"]["resume_checkpoint"] = ""
            trial_out = output_dir / "trials" / f"trial_{idx:03d}"
            trial_cfg["runtime"]["output_dir"] = str(trial_out)
            for k, v in combo.items():
                _set_by_dotted_key(trial_cfg, k, v)

            print(f"[Trial {idx}/{len(combos)}] {combo}")
            result = _train_once(trial_cfg)
            trial_item = {
                "trial_index": idx,
                "overrides": combo,
                "result": result,
            }
            all_results.append(trial_item)

            cur_miou = float(result["best_metrics"]["best_val_miou"])
            if cur_miou > best_miou:
                best_miou = cur_miou
                best_trial = trial_item

        search_summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_trials": len(all_results),
            "best_trial": best_trial,
            "trials": all_results,
        }
        _write_json(output_dir / "search_results.json", search_summary)

        if best_trial is not None:
            best_hparams = best_trial["overrides"]
            _write_json(output_dir / "best_hparams.json", best_hparams)

        print(f"Search completed. Best mIoU={best_miou:.4f}")
        return

    result = _train_once(base_cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_eval_command(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    base_cfg = _materialize_run_config(cfg, args)

    runtime_cfg = base_cfg["runtime"]
    output_dir = Path(_as_str(runtime_cfg, "output_dir", ""))
    _ensure_dir(output_dir)

    ckpt_path = Path(args.checkpoint)
    ckpt = _load_checkpoint(ckpt_path, device=_select_device(_as_str(runtime_cfg, "device", "")))

    model_state = ckpt.get("model_state_dict", ckpt)

    train_ds, val_ds, _, _ = _build_train_val_datasets(base_cfg)
    _ = train_ds

    sample_x, _ = val_ds[0]
    in_channels_cfg = int(sample_x.shape[0])
    in_channels = int(ckpt.get("in_channels", 0))
    if in_channels <= 0:
        # Fallback: infer from first conv weight in checkpoint.
        first_key = None
        for key in ("enc1.net.0.weight", "stem.conv.weight", "enc1.block1.conv1.weight"):
            if key in model_state:
                first_key = key
                break
        if first_key is None:
            # Last resort: use config-derived channels.
            in_channels = in_channels_cfg
        else:
            in_channels = int(model_state[first_key].shape[1])

    if in_channels_cfg != in_channels:
        raise ValueError(
            "Input channel mismatch between dataset and checkpoint. "
            f"dataset_channels={in_channels_cfg}, checkpoint_channels={in_channels}. "
            "Please use the config that matches the checkpoint training setup (e.g., embedding_key, "
            "embeddings_per_time vs embeddings, resample_size), or point to the correct checkpoint."
        )

    num_classes_cfg = _as_int(base_cfg["labels"], "num_classes", 9)
    model_meta = ckpt.get("model_cfg", None)
    if model_meta is None:
        inferred = _infer_model_meta_from_state(model_state)
        model_cfg = _merge_dict(base_cfg.get("model", {}), {
            "variant": inferred["variant"],
            "depth": inferred["depth"],
            "base_channels": inferred["base_channels"],
        })
        num_classes = int(inferred["num_classes"])
    else:
        model_cfg = _merge_dict(base_cfg.get("model", {}), model_meta)
        num_classes = int(ckpt.get("num_classes", num_classes_cfg))

    device = _select_device(_as_str(runtime_cfg, "device", ""))

    model = _build_model(
        in_channels=in_channels,
        num_classes=num_classes,
        model_cfg=model_cfg,
    ).to(device)

    model.load_state_dict(model_state, strict=_as_bool(runtime_cfg, "resume_strict", True))

    labels_cfg = base_cfg["labels"]
    ignore_index = _as_int(labels_cfg, "ignore_index", 255)
    background_index = _as_int(labels_cfg, "background_index", 0)
    ignore_bg_metrics = _as_bool(labels_cfg, "ignore_background_in_metrics", True)
    ignore_bg_dice = _as_bool(labels_cfg, "ignore_background_in_dice", True)

    train_cfg = base_cfg["train"]
    criterion_focal = FocalLossWithSampling(
        num_classes=num_classes,
        gamma=_as_float(train_cfg, "focal_gamma", 2.0),
        ignore_index=ignore_index,
        focus_classes=_parse_focus_classes(
            _as_str(train_cfg, "focal_focus_classes", "all_fg"),
            num_classes,
            background_index,
            ignore_index,
        ),
        class_weights=_parse_float_list(train_cfg.get("focal_class_weights")),
    )
    criterion_dice = SoftDiceLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background=ignore_bg_dice,
    )
    ce_weight = _as_float(train_cfg, "ce_weight", 0.0)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if ce_weight > 0.0 else None

    val_loader = _build_loader(
        val_ds,
        batch_size=_as_int(train_cfg, "batch_size", 4),
        num_workers=_as_int(train_cfg, "num_workers", 4),
        device=device,
        shuffle=False,
    )

    stats = _run_one_epoch(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background_in_metrics=ignore_bg_metrics,
        criterion_focal=criterion_focal,
        criterion_dice=criterion_dice,
        criterion_ce=criterion_ce,
        fg_aux_weight=_as_float(train_cfg, "fg_aux_weight", 0.0),
        bg_ratio_weight=_as_float(train_cfg, "bg_ratio_weight", 0.0),
        dice_weight=_as_float(train_cfg, "dice_weight", 0.4),
        ce_weight=ce_weight,
        optimizer=None,
        amp_enabled=bool(device.type == "cuda" and _as_bool(runtime_cfg, "amp", True)),
        grad_scaler=None,
        use_channels_last=bool(device.type == "cuda" and _as_bool(runtime_cfg, "channels_last", True)),
    )

    per_class_iou = evaluate_per_class_iou(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
        background_index=background_index,
        ignore_background=ignore_bg_metrics,
    )

    eval_out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(ckpt_path),
        "input_mode": _as_str(base_cfg["input"], "mode", "embedding"),
        "metrics": {
            "val_loss": float(stats["loss"]),
            "val_acc": float(stats["acc"]),
            "val_miou": float(stats["miou"]),
            "val_f1": float(stats["f1"]),
            "val_mf1": float(stats["mf1"]),
            "val_precision": float(stats["precision"]),
            "val_recall": float(stats["recall"]),
            "val_pred_bg_ratio": float(stats["pred_bg_ratio"]),
        },
        "num_classes": num_classes,
    }

    _write_json(output_dir / "eval_result.json", eval_out)
    dump_per_class_iou_report(
        per_class_iou,
        output_dir=output_dir,
        prefix="eval",
        per_class_f1=stats["per_class_f1"],
        weighted_f1=stats["weighted_f1"],
    )
    save_confusion_matrix_visualization(
        stats["confusion_matrix"],
        output_dir=output_dir,
        prefix="eval",
        class_names=[str(i) for i in range(num_classes)],
        normalize=True,
    )

    if _as_int(train_cfg, "save_val_samples", 6) > 0:
        save_val_visualizations(
            model=model,
            val_ds=val_ds,
            device=device,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_index=ignore_index,
            max_images=_as_int(train_cfg, "save_val_samples", 6),
        )

    print(json.dumps(eval_out, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified segmentation training/evaluation for embedding or raw input modes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--config", type=str, required=True, help="YAML/JSON config path")
    p_train.add_argument("--output_dir", type=str, default="", help="Override runtime.output_dir")
    p_train.add_argument("--device", type=str, default="", help="Override runtime.device")
    p_train.add_argument("--resume", action="store_true", help="Resume from latest/checkpoint")
    p_train.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Explicit checkpoint path for resume (default: output_dir/checkpoints/latest.pt)",
    )

    p_eval = sub.add_parser("eval", help="Run evaluation with checkpoint")
    p_eval.add_argument("--config", type=str, required=True, help="YAML/JSON config path")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    p_eval.add_argument("--output_dir", type=str, default="", help="Override runtime.output_dir")
    p_eval.add_argument("--device", type=str, default="", help="Override runtime.device")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        _run_train_command(args)
    elif args.command == "eval":
        _run_eval_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
