import argparse
import csv
import copy
import importlib
import importlib.util
import json
import re
import pickle
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    import yaml
except ImportError:
    yaml = None

def _bootstrap_scipy_propack_compat() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        return
    try:
        scipy_root = Path(scipy.__file__).resolve().parent
        propack_matches = sorted((scipy_root / "sparse" / "linalg").glob("_propack*.so"))
        if not propack_matches:
            return
        propack_path = propack_matches[0]
        existing = sys.modules.get("scipy.sparse.linalg._propack")
        if existing is not None:
            for alias in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
                if not hasattr(existing, alias):
                    setattr(existing, alias, existing)
            return
        spec = importlib.util.spec_from_file_location("scipy.sparse.linalg._propack", propack_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for alias in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
            if not hasattr(module, alias):
                setattr(module, alias, module)
        sys.modules["scipy.sparse.linalg._propack"] = module
    except Exception:
        return


try:
    _bootstrap_scipy_propack_compat()
    from sklearn.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover - dependency issue
    raise ImportError(
        "scikit-learn is required for Random Forest training. Please install scikit-learn first."
    ) from exc

try:
    import cupy as cp
except ImportError:
    cp = None

def _load_cuml_random_forest_classifier():
    if cp is None:
        return None
    try:
        propack = importlib.import_module("scipy.sparse.linalg._propack")
        if not hasattr(propack, "_spropack"):
            setattr(propack, "_spropack", propack)
    except Exception:
        pass
    try:
        from cuml.ensemble import RandomForestClassifier as cuml_random_forest_classifier
    except ImportError:
        return None
    return cuml_random_forest_classifier


CuMLRandomForestClassifier = _load_cuml_random_forest_classifier()

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


RF_RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
RF_MIN_SAMPLES_LEAF = 1
RF_BOOTSTRAP = True
RF_MAX_SAMPLES = 1.0


def _gpu_rf_available() -> bool:
    return cp is not None and CuMLRandomForestClassifier is not None and torch.cuda.is_available()


def _torch_to_cupy(tensor: torch.Tensor):
    if cp is None:
        raise ImportError("cupy is required for GPU Random Forest backend")
    return cp.asarray(tensor.detach().cpu().numpy())


def _load_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
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


def _default_train_config() -> Dict[str, Any]:
    return {
        "input": {"mode": "embedding"},
        "labels": {
            "num_classes": 20,
            "ignore_index": 255,
            "background_index": 0,
            "ignore_background_in_metrics": True,
        },
        "train": {
            "seed": RF_RANDOM_STATE,
        },
        "rf": {
            "backend": "auto",
            "n_estimators": RF_N_ESTIMATORS,
            "max_features_mode": "input_channels",
            "max_features_value": 64,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "bootstrap": RF_BOOTSTRAP,
            "max_samples": RF_MAX_SAMPLES,
            "random_state": RF_RANDOM_STATE,
            "n_jobs": -1,
            "verbose": 0,
        },
        "data": {
            "resample_size": 0,
            "val_fraction": 0.2,
            "max_train_pixels_per_image": 0,
            "max_train_pixels_total": 0,
            "augment_repeats": 0,
            "augment_hflip_prob": 0.5,
            "augment_vflip_prob": 0.5,
            "augment_rot90": True,
            "augment_noise_std": 0.0,
            "balance_minority_pixels": False,
            "balance_max_multiplier": 4.0,
            "balance_include_background": False,
            "save_val_samples": 4,
        },
        "runtime": {
            "device": "cpu",
            "output_dir": "",
        },
        "embedding": {
            "embedding_key": "auto",
            "per_patch_labels": False,
            "train_embeddings_path": "",
            "train_labels_path": "",
            "val_embeddings_path": "",
            "val_labels_path": "",
        },
        "raw": {
            "dataset_root": "",
            "label_root": "",
            "train_split": "train",
            "val_split": "val",
            "s1_key": "sentinel1",
            "s2_key": "sentinel2",
            "label_key": "labels",
            "time_steps": 12,
        },
    }


def _materialize_run_config(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    merged = _merge_dict(_default_train_config(), cfg)

    runtime = merged.setdefault("runtime", {})
    if getattr(args, "output_dir", ""):
        runtime["output_dir"] = args.output_dir
    elif output_dir_override is not None:
        runtime["output_dir"] = output_dir_override

    if getattr(args, "device", ""):
        runtime["device"] = args.device

    if not runtime.get("output_dir"):
        raise ValueError("runtime.output_dir is required (or use --output_dir).")

    return merged


def _namespace_from_config(cfg: Dict[str, Any]) -> argparse.Namespace:
    input_cfg = cfg.get("input", {})
    labels_cfg = cfg.get("labels", {})
    rf_cfg = cfg.get("rf", {})
    data_cfg = cfg.get("data", {})
    runtime_cfg = cfg.get("runtime", {})
    embedding_cfg = cfg.get("embedding", {})
    raw_cfg = cfg.get("raw", {})
    train_cfg = cfg.get("train", {})

    return argparse.Namespace(
        input_mode=_as_str(input_cfg, "mode", "embedding"),
        output_dir=_as_str(runtime_cfg, "output_dir", ""),
        device=_as_str(runtime_cfg, "device", "cpu"),
        seed=_as_int(train_cfg, "seed", RF_RANDOM_STATE),
        num_classes=_as_int(labels_cfg, "num_classes", 20),
        ignore_index=_as_int(labels_cfg, "ignore_index", 255),
        background_index=_as_int(labels_cfg, "background_index", 0),
        ignore_background_in_metrics=_as_bool(labels_cfg, "ignore_background_in_metrics", True),
        embeddings_path=_as_str(embedding_cfg, "train_embeddings_path", ""),
        labels_file=_as_str(embedding_cfg, "train_labels_path", ""),
        val_embeddings_path=_as_str(embedding_cfg, "val_embeddings_path", ""),
        val_labels_file=_as_str(embedding_cfg, "val_labels_path", ""),
        embedding_key=_as_str(embedding_cfg, "embedding_key", "auto"),
        resample_size=_as_int(data_cfg, "resample_size", 0),
        per_patch_labels=_as_bool(embedding_cfg, "per_patch_labels", False),
        val_fraction=_as_float(data_cfg, "val_fraction", 0.2),
        dataset_root=_as_str(raw_cfg, "dataset_root", ""),
        label_root=_as_str(raw_cfg, "label_root", ""),
        train_split=_as_str(raw_cfg, "train_split", "train"),
        val_split=_as_str(raw_cfg, "val_split", "val"),
        s1_key=_as_str(raw_cfg, "s1_key", "sentinel1"),
        s2_key=_as_str(raw_cfg, "s2_key", "sentinel2"),
        label_key=_as_str(raw_cfg, "label_key", "labels"),
        time_steps=_as_int(raw_cfg, "time_steps", 12),
        max_train_pixels_per_image=_as_int(data_cfg, "max_train_pixels_per_image", 0),
        max_train_pixels_total=_as_int(data_cfg, "max_train_pixels_total", 0),
        augment_repeats=_as_int(data_cfg, "augment_repeats", 0),
        augment_hflip_prob=_as_float(data_cfg, "augment_hflip_prob", 0.5),
        augment_vflip_prob=_as_float(data_cfg, "augment_vflip_prob", 0.5),
        augment_rot90=_as_bool(data_cfg, "augment_rot90", True),
        augment_noise_std=_as_float(data_cfg, "augment_noise_std", 0.0),
        balance_minority_pixels=_as_bool(data_cfg, "balance_minority_pixels", False),
        balance_max_multiplier=_as_float(data_cfg, "balance_max_multiplier", 4.0),
        balance_include_background=_as_bool(data_cfg, "balance_include_background", False),
        save_val_samples=_as_int(data_cfg, "save_val_samples", 4),
        n_jobs=_as_int(rf_cfg, "n_jobs", -1),
        verbose=_as_int(rf_cfg, "verbose", 0),
        max_features_mode=_as_str(rf_cfg, "max_features_mode", "input_channels"),
        max_features_value=_as_int(rf_cfg, "max_features_value", 64),
        rf_n_estimators=_as_int(rf_cfg, "n_estimators", RF_N_ESTIMATORS),
        rf_backend=_as_str(rf_cfg, "backend", "auto"),
        rf_min_samples_leaf=_as_int(rf_cfg, "min_samples_leaf", RF_MIN_SAMPLES_LEAF),
        rf_bootstrap=_as_bool(rf_cfg, "bootstrap", RF_BOOTSTRAP),
        rf_max_samples=_as_float(rf_cfg, "max_samples", RF_MAX_SAMPLES),
        rf_random_state=_as_int(rf_cfg, "random_state", RF_RANDOM_STATE),
    )


def prepare_features_from_embeddings(emb_npy: Path, embedding_key: str = "auto") -> np.ndarray:
    data = np.load(emb_npy, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()

    if isinstance(data, np.ndarray):
        e = data
        if e.ndim == 4:
            if e.shape[-1] == 64:
                e_chw = np.transpose(e, (0, 3, 1, 2))
            elif e.shape[1] == 64:
                e_chw = e
            else:
                raise ValueError(f"Unexpected ndarray embedding shape {e.shape} in {emb_npy}")
            t, c, h, w = e_chw.shape
            feats = e_chw.reshape(t * c, h, w)
        elif e.ndim == 3:
            if e.shape[-1] == 64:
                feats = np.transpose(e, (2, 0, 1))
            else:
                feats = e
        else:
            raise ValueError(f"Unsupported ndarray embedding ndim={e.ndim} in {emb_npy}")
        return feats.astype(np.float32)

    if isinstance(data, dict):
        if embedding_key != "auto":
            if embedding_key not in data:
                raise ValueError(f"Embedding key '{embedding_key}' not found in {emb_npy}")
            e = data[embedding_key]
            if e.ndim == 4:
                if e.shape[-1] != 64:
                    raise ValueError(f"Unexpected {embedding_key} shape {e.shape} in {emb_npy}")
                e_chw = np.transpose(e, (0, 3, 1, 2))
                t, c, h, w = e_chw.shape
                feats = e_chw.reshape(t * c, h, w)
            elif e.ndim == 3:
                if e.shape[-1] != 64:
                    raise ValueError(f"Unexpected {embedding_key} shape {e.shape} in {emb_npy}")
                feats = np.transpose(e, (2, 0, 1))
            else:
                raise ValueError(f"Unsupported {embedding_key} ndim={e.ndim} in {emb_npy}")
        elif "embeddings_per_time" in data:
            e = data["embeddings_per_time"]
            if e.ndim != 4 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings_per_time shape {e.shape} in {emb_npy}")
            e_chw = np.transpose(e, (0, 3, 1, 2))
            t, c, h, w = e_chw.shape
            feats = e_chw.reshape(t * c, h, w)
        elif "embeddings" in data:
            e = data["embeddings"]
            if e.ndim != 3 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings shape {e.shape} in {emb_npy}")
            feats = np.transpose(e, (2, 0, 1))
        elif "embeddings_native" in data:
            e = data["embeddings_native"]
            if e.ndim != 3 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings_native shape {e.shape} in {emb_npy}")
            feats = np.transpose(e, (2, 0, 1))
        else:
            raise ValueError(f"None of 'embeddings_per_time'/'embeddings'/'embeddings_native' found in {emb_npy}")
    else:
        raise ValueError(f"Unsupported embeddings payload type in {emb_npy}: {type(data)}")

    feats = feats.astype(np.float32, copy=False)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def resize_features_to(features: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if features.shape[-2:] == (out_h, out_w):
        return features
    t = torch.from_numpy(features).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)


def load_label_mask_from_file(label_file: Path, label_key: str = "labels") -> np.ndarray:
    if label_file.suffix.lower() == ".npz":
        with np.load(label_file, allow_pickle=True) as label_npz:
            if label_key in label_npz:
                labels = label_npz[label_key]
            elif "labels" in label_npz:
                labels = label_npz["labels"]
            else:
                available = list(label_npz.keys())
                if len(available) == 1:
                    labels = label_npz[available[0]]
                else:
                    raise KeyError(f"Label key '{label_key}' not found in {label_file}. Available keys: {available}")
    else:
        labels_raw = np.load(label_file, allow_pickle=True)
        if isinstance(labels_raw, np.ndarray) and labels_raw.dtype == object and labels_raw.shape == ():
            labels_obj = labels_raw.item()
            if isinstance(labels_obj, dict):
                if label_key in labels_obj:
                    labels = labels_obj[label_key]
                elif "labels" in labels_obj:
                    labels = labels_obj["labels"]
                else:
                    available = list(labels_obj.keys())
                    if len(available) == 1:
                        labels = labels_obj[available[0]]
                    else:
                        raise KeyError(
                            f"Label key '{label_key}' not found in {label_file}. Available keys: {available}"
                        )
            else:
                labels = labels_obj
        else:
            labels = labels_raw

    labels_np = np.asarray(labels).squeeze()
    if labels_np.ndim != 2:
        raise ValueError(f"Expected 2D label in {label_file}, got shape {labels_np.shape}")
    return labels_np


def resize_labels_to(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if features.shape[-2:] == labels.shape:
        return labels
    t = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=features.shape[-2:], mode="nearest")
    return t.squeeze(0).squeeze(0).numpy().astype(labels.dtype)


def estimate_label_valid_ratio(
    dataset: Dataset,
    num_classes: int,
    ignore_index: int,
    max_samples: int = 16,
) -> Tuple[float, int, int]:
    n = min(len(dataset), max(1, int(max_samples)))
    valid_count = 0
    total_count = 0
    global_min = None
    global_max = None

    for i in range(n):
        _x, y = dataset[i]
        y = y.reshape(-1).to(torch.int64)
        ymin = int(y.min().item())
        ymax = int(y.max().item())
        global_min = ymin if global_min is None else min(global_min, ymin)
        global_max = ymax if global_max is None else max(global_max, ymax)

        valid = (y == ignore_index) | ((y >= 0) & (y < num_classes))
        valid_count += int(valid.sum().item())
        total_count += int(y.numel())

    ratio = float(valid_count) / float(max(1, total_count))
    return ratio, int(global_min if global_min is not None else 0), int(global_max if global_max is not None else 0)


def labels_to_rgb(mask: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    if plt is None:
        raise ImportError("matplotlib is not available in this environment")
    cmap = plt.get_cmap("tab20", num_classes)
    palette = cmap(np.arange(num_classes))[:, :3]
    safe_mask = np.clip(mask, 0, num_classes - 1)
    rgb = palette[safe_mask]
    rgb[mask == ignore_index] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return rgb


def dump_per_class_iou_report(
    per_class_iou: np.ndarray,
    output_dir: Path,
    prefix: str = "best_val",
    per_class_f1: Optional[np.ndarray] = None,
    weighted_f1: Optional[float] = None,
) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{prefix}_per_class_iou.npy"
    json_path = output_dir / f"{prefix}_per_class_iou.json"
    csv_path = output_dir / f"{prefix}_per_class_iou.csv"

    np.save(npy_path, per_class_iou)

    rows: List[dict] = []
    print("Per-class IoU report (NaN = ignored classes; IoU=0.0 indicates no pixels):")
    for cls_idx, iou in enumerate(per_class_iou):
        iou_value = None if np.isnan(iou) else float(iou)
        row = {"class_index": int(cls_idx), "iou": iou_value}
        if per_class_f1 is not None and cls_idx < len(per_class_f1):
            f1_value = None if np.isnan(per_class_f1[cls_idx]) else float(per_class_f1[cls_idx])
            row["f1"] = f1_value
        if weighted_f1 is not None:
            row["weighted_f1"] = None if np.isnan(weighted_f1) else float(weighted_f1)
        rows.append(row)
        if iou_value is None:
            if "f1" in row:
                f1_text = "NaN" if row["f1"] is None else f"{row['f1']:.4f}"
                wf1_text = "NaN" if row.get("weighted_f1") is None else f"{row['weighted_f1']:.4f}"
                print(f"  class {cls_idx}: IoU=NaN, F1={f1_text}, weighted_F1={wf1_text}")
            else:
                print(f"  class {cls_idx}: NaN")
        else:
            if "f1" in row:
                f1_text = "NaN" if row["f1"] is None else f"{row['f1']:.4f}"
                wf1_text = "NaN" if row.get("weighted_f1") is None else f"{row['weighted_f1']:.4f}"
                print(f"  class {cls_idx}: IoU={iou_value:.4f}, F1={f1_text}, weighted_F1={wf1_text}")
            else:
                print(f"  class {cls_idx}: {iou_value:.4f}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["class_index", "iou"]
        if per_class_f1 is not None:
            header.append("f1")
        if weighted_f1 is not None:
            header.append("weighted_f1")
        writer.writerow(header)
        for row in rows:
            csv_row = [row["class_index"], "" if row["iou"] is None else f"{row['iou']:.6f}"]
            if per_class_f1 is not None:
                csv_row.append("" if row.get("f1") is None else f"{row['f1']:.6f}")
            if weighted_f1 is not None:
                csv_row.append("" if row.get("weighted_f1") is None else f"{row['weighted_f1']:.6f}")
            writer.writerow(csv_row)

    print(f"Saved per-class IoU report to {npy_path}")
    print(f"Saved per-class IoU report to {json_path}")
    print(f"Saved per-class IoU report to {csv_path}")
    return rows


def save_confusion_matrix_visualization(
    confusion_matrix: np.ndarray,
    output_dir: Path,
    prefix: str = "val",
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape {cm.shape}")

    np.save(output_dir / f"{prefix}_confusion_matrix.npy", cm.astype(np.int64))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    np.save(output_dir / f"{prefix}_confusion_matrix_norm.npy", cm_norm.astype(np.float32))

    if plt is None:
        print("matplotlib is not available; skipping confusion matrix PNG visualization.")
        return

    plot_cm = cm_norm if normalize else cm
    fig_size = max(8.0, 0.35 * cm.shape[0] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(plot_cm, cmap="Blues", interpolation="nearest", vmin=0.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Validation confusion matrix" + (" (row-normalized)" if normalize else ""))

    if cm.shape[0] <= 20:
        thresh = float(np.nanmax(plot_cm)) * 0.6 if plot_cm.size > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = plot_cm[i, j]
                text = f"{value:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    candidates = [root / f"{split}_labels_npz", root / f"{split}_labels", root / split, root]
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
    x = _to_4d(arr, key=key, path=path)
    t, a, b, c = x.shape
    if c <= 64 and a > 64 and b > 64:
        return x
    if a <= 64 and b > 64 and c > 64:
        if t >= a:
            return np.transpose(x, (0, 2, 3, 1))
        return np.transpose(x, (1, 2, 3, 0))
    if t <= 64 and a <= 64 and b > 64 and c > 64:
        return np.transpose(x, (1, 2, 3, 0))
    return x


def _normalize_source(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    max_abs = float(np.nanmax(np.abs(x))) if x.size > 0 else 0.0
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
    t, h, w, c = arr.shape
    chw = np.transpose(arr, (0, 3, 1, 2))
    return chw.reshape(t * c, h, w).astype(np.float32)


def _augment_segmentation_pair(
    x_np: np.ndarray,
    y_np: np.ndarray,
    rng: np.random.Generator,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rot90: bool = True,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x_aug = np.asarray(x_np, dtype=np.float32)
    y_aug = np.asarray(y_np)

    if rot90:
        k = int(rng.integers(0, 4))
        if k:
            x_aug = np.rot90(x_aug, k=k, axes=(1, 2))
            y_aug = np.rot90(y_aug, k=k, axes=(0, 1))

    if rng.random() < float(hflip_prob):
        x_aug = np.flip(x_aug, axis=2)
        y_aug = np.flip(y_aug, axis=1)

    if rng.random() < float(vflip_prob):
        x_aug = np.flip(x_aug, axis=1)
        y_aug = np.flip(y_aug, axis=0)

    if noise_std > 0:
        noise = rng.normal(0.0, float(noise_std), size=x_aug.shape).astype(np.float32)
        x_aug = x_aug + noise

    return np.ascontiguousarray(x_aug, dtype=np.float32), np.ascontiguousarray(y_aug)


class EmbeddingSegmentationDataset(Dataset):
    def __init__(
        self,
        embeddings_path: Path,
        labels_path: Path,
        per_patch_labels: bool = False,
        embedding_key: str = "auto",
        resample_size: int = 0,
    ):
        self.embeddings_path = embeddings_path
        self.per_patch_labels = per_patch_labels
        self.embedding_key = embedding_key
        self.resample_size = max(0, int(resample_size))
        self.index_label_files: Optional[List[Path]] = None

        if per_patch_labels:
            if not labels_path.is_dir():
                raise ValueError(f"With per_patch_labels=True, labels_path must be a directory, got {labels_path}")
            self.labels_dir = labels_path
            self.labels_np = None
        else:
            if labels_path.suffix.lower() != ".npy":
                raise ValueError(f"Labels must be a .npy file, got {labels_path}")
            labels = np.load(labels_path, allow_pickle=True)
            if labels.ndim != 2:
                raise ValueError(f"Expected labels with shape (H,W), got {labels.shape}")
            self.labels_np = labels
            self.labels_dir = None

        if embeddings_path.is_dir():
            self.files = sorted([p for p in embeddings_path.glob("*.npy") if p.is_file()])
            if not self.files:
                raise FileNotFoundError(f"No embedding files (.npy) found in embeddings directory {embeddings_path}")
        else:
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings npy not found: {embeddings_path}")
            if embeddings_path.suffix.lower() != ".npy":
                raise ValueError(f"Embeddings must be a .npy file, got {embeddings_path}")
            self.files = [embeddings_path]

        if self.per_patch_labels and self.labels_dir is not None and embeddings_path.is_dir():
            label_files = sorted([p for p in self.labels_dir.glob("*_label.npy") if p.is_file()])
            has_sample_prefix = any(p.name.startswith("sample_") for p in label_files)
            has_parcel_prefix = any(p.name.startswith("ParcelIDs_") for p in label_files)
            if (not has_sample_prefix and not has_parcel_prefix) and len(label_files) == len(self.files):
                self.index_label_files = label_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_file = self.files[idx]
        feats = prepare_features_from_embeddings(emb_file, embedding_key=self.embedding_key)
        if self.resample_size > 0:
            feats = resize_features_to(feats, self.resample_size, self.resample_size)

        if self.per_patch_labels:
            label_file: Optional[Path] = None
            candidate_paths: List[Path] = []
            if self.index_label_files is not None:
                if idx >= len(self.index_label_files):
                    raise IndexError(f"Index {idx} out of range for index-based labels list of length {len(self.index_label_files)}")
                label_file = self.index_label_files[idx]
            else:
                stem = emb_file.stem
                patch_token = stem[len("embedding_") :] if stem.startswith("embedding_") else stem
                if self.labels_dir is not None:
                    candidate_paths.append(self.labels_dir / f"sample_{patch_token}_label.npy")
                    m_eopath = re.match(r"^eopath_(\d+)_\d+_\d+$", patch_token)
                    if m_eopath is not None:
                        patch_id = int(m_eopath.group(1))
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id:05d}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id:05d}.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id}.npy")
                    m = re.search(r"(\d+)$", patch_token)
                    if m is not None:
                        patch_suffix = m.group(1)
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{int(patch_suffix):05d}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_suffix}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{int(patch_suffix):05d}.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_suffix}.npy")

                for cand in candidate_paths:
                    if cand.exists():
                        label_file = cand
                        break

            if label_file is None:
                raise FileNotFoundError(
                    "Per-patch label file not found for embedding "
                    f"{emb_file.name}; tried: " + ", ".join(str(c) for c in candidate_paths)
                )

            labels_np = load_label_mask_from_file(label_file)
        else:
            labels_np = self.labels_np  # type: ignore[assignment]

        labels_np = labels_np.astype(np.int64)
        labels_resized = resize_labels_to(feats, labels_np)
        features = torch.from_numpy(feats)
        labels = torch.from_numpy(labels_resized.astype(np.int64))
        return features, labels


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
            [p for ext in ("sample_*.npz", "sample_*.npy", "*_image.npz", "*_image.npy") for p in raw_dir.glob(ext) if p.is_file()]
        )
        lbl_files = sorted(
            [
                p
                for ext in ("ParcelIDs_*_labels.npz", "ParcelIDs_*_labels.npy", "ParcelIDs_*.npy", "*_label.npz", "*_label.npy")
                for p in lbl_dir.glob(ext)
                if p.is_file()
            ]
        )
        if not raw_files:
            raw_files = sorted(
                [p for ext in ("*.npz", "*.npy") for p in raw_dir.glob(ext) if p.is_file() and not p.name.endswith("_label.npy") and not p.name.endswith("_label.npz")]
            )
        if not lbl_files:
            lbl_files = sorted([p for ext in ("*.npz", "*.npy") for p in lbl_dir.glob(ext) if p.is_file() and not p.name.startswith("sample_")])

        if not raw_files:
            raise FileNotFoundError(f"No raw files found in {raw_dir}. Expected patterns like sample_*.npz or *_image.npy.")
        if not lbl_files:
            raise FileNotFoundError(f"No label files found in {lbl_dir}")

        raw_map = _build_id_file_map(raw_files)
        lbl_map = _build_id_file_map(lbl_files)
        common_ids = sorted(set(raw_map.keys()) & set(lbl_map.keys()), key=_sort_patch_id)
        if not common_ids:
            raise RuntimeError(f"No matched patch IDs between raw ({raw_dir}) and labels ({lbl_dir}).")

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
                raise TypeError(f"Unsupported object npy payload in {raw_file}: expected dict, got {type(obj)}")
            raw_container = obj
        elif isinstance(raw_data, np.ndarray):
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
                raise KeyError(f"Missing source key(s) in {raw_file}. Expected {missing}. Available keys: {available}")

            s1 = _to_thwc(raw_container[s1_key], key=s1_key, path=raw_file) if s1_key else None
            s2 = _to_thwc(raw_container[s2_key], key=s2_key, path=raw_file)
        else:
            if raw_array is None:
                raise RuntimeError("Internal error: raw_container and raw_array are both None")
            if raw_array.ndim == 4:
                s2 = _to_thwc(raw_array, key="raw", path=raw_file)
            elif raw_array.ndim == 3:
                if raw_array.shape[0] == self.time_steps:
                    s2 = raw_array[:, :, :, None]
                else:
                    s2 = raw_array[None, :, :, :]
                s2 = _to_thwc(s2, key="raw", path=raw_file)
            else:
                raise ValueError(f"Unsupported ndarray raw shape in {raw_file}: expected 3D/4D, got {raw_array.shape}")
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
                        raise KeyError(f"Label key '{self.label_key}' not found in {lbl_file}. Available keys: {available}")
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
                            raise KeyError(f"Label key '{self.label_key}' not found in {lbl_file}. Available keys: {available}")
                else:
                    y = y_obj
            else:
                y = y_raw

        y = np.asarray(y).squeeze().astype(np.int64)
        if y.ndim != 2:
            raise ValueError(f"Expected 2D label in {lbl_file}, got shape {y.shape}")

        if self.resample_size > 0:
            x_chw = resize_features_to(x_chw, self.resample_size, self.resample_size)

        y = resize_labels_to(x_chw, y)
        return torch.from_numpy(x_chw), torch.from_numpy(y.astype(np.int64))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_embedding_dataset(args: argparse.Namespace, split: str) -> EmbeddingSegmentationDataset:
    if split == "train":
        emb_path = Path(args.embeddings_path)
        labels_path = Path(args.labels_file)
    else:
        emb_path = Path(args.val_embeddings_path)
        labels_path = Path(args.val_labels_file)

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {emb_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path not found: {labels_path}")

    per_patch_labels = bool(args.per_patch_labels or labels_path.is_dir())
    if split == "train" and labels_path.is_dir() and not args.per_patch_labels:
        print("Detected train labels directory; enabling per-patch label matching automatically.")
    if split == "val" and labels_path.is_dir() and not args.per_patch_labels:
        print("Detected val labels directory; enabling per-patch label matching automatically.")

    return EmbeddingSegmentationDataset(
        embeddings_path=emb_path,
        labels_path=labels_path,
        per_patch_labels=per_patch_labels,
        embedding_key=args.embedding_key,
        resample_size=args.resample_size,
    )


def _build_raw_dataset(args: argparse.Namespace, split: str) -> PastisRawSegmentationDataset:
    dataset_root = Path(args.dataset_root)
    label_root = Path(args.label_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label_root not found: {label_root}")

    split_name = args.train_split if split == "train" else args.val_split
    return PastisRawSegmentationDataset(
        dataset_root=dataset_root,
        label_root=label_root,
        split=split_name,
        s1_key=args.s1_key,
        s2_key=args.s2_key,
        label_key=args.label_key,
        time_steps=args.time_steps,
        resample_size=args.resample_size,
    )


def _build_train_val_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, Dataset]:
    if args.input_mode == "embedding":
        train_base = _build_embedding_dataset(args, "train")
        if args.val_embeddings_path and args.val_labels_file:
            val_base = _build_embedding_dataset(args, "val")
        else:
            if len(train_base) < 2:
                raise ValueError("Embedding train dataset has fewer than 2 samples; cannot split train/val.")
            rng = np.random.default_rng(args.seed)
            n_total = len(train_base)
            n_val = max(1, int(n_total * args.val_fraction)) if n_total > 1 else 0
            n_train = n_total - n_val
            if n_train <= 0:
                raise ValueError("val_fraction leaves no training samples.")
            indices = rng.permutation(n_total)
            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train:].tolist()
            val_base = Subset(_build_embedding_dataset(args, "train"), val_idx)
            train_base = Subset(train_base, train_idx)
        return train_base, val_base, train_base

    if args.input_mode == "raw":
        train_base = _build_raw_dataset(args, "train")
        val_base = _build_raw_dataset(args, "val")
        if len(train_base) == 0 or len(val_base) == 0:
            raise ValueError("Raw train or val dataset is empty.")
        return train_base, val_base, train_base

    raise ValueError(f"Unsupported input_mode: {args.input_mode}")


def _sample_pixels_from_item(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    max_pixels_per_image: int,
    rng: np.random.Generator,
    tensor_device: Optional[torch.device] = None,
    use_gpu_backend: bool = False,
    augment: bool = False,
    augment_hflip_prob: float = 0.5,
    augment_vflip_prob: float = 0.5,
    augment_rot90: bool = True,
    augment_noise_std: float = 0.0,
) -> Tuple[Any, Any]:
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    y_np = y.detach().cpu().numpy().astype(np.int64, copy=False)

    if augment:
        x_np, y_np = _augment_segmentation_pair(
            x_np,
            y_np,
            rng=rng,
            hflip_prob=augment_hflip_prob,
            vflip_prob=augment_vflip_prob,
            rot90=augment_rot90,
            noise_std=augment_noise_std,
        )

    if x_np.ndim != 3:
        raise ValueError(f"Expected 3D feature tensor (C,H,W), got shape {x_np.shape}")
    if y_np.ndim != 2:
        raise ValueError(f"Expected 2D label tensor (H,W), got shape {y_np.shape}")

    c, _, _ = x_np.shape
    x_flat = np.moveaxis(x_np, 0, -1).reshape(-1, c)
    y_flat = y_np.reshape(-1)
    valid = (y_flat != ignore_index) & (y_flat >= 0) & (y_flat < num_classes)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        x_empty = torch.empty((0, c), dtype=torch.float32)
        y_empty = torch.empty((0,), dtype=torch.int64)
        if use_gpu_backend:
            return _torch_to_cupy(x_empty), _torch_to_cupy(y_empty)
        return x_empty.numpy(), y_empty.numpy()

    if max_pixels_per_image > 0 and valid_idx.size > max_pixels_per_image:
        valid_idx = rng.choice(valid_idx, size=max_pixels_per_image, replace=False)

    x_sel = torch.from_numpy(x_flat[valid_idx].astype(np.float32, copy=False))
    y_sel = torch.from_numpy(y_flat[valid_idx].astype(np.int64, copy=False))

    if use_gpu_backend:
        return _torch_to_cupy(x_sel), _torch_to_cupy(y_sel)
    return x_sel.numpy(), y_sel.numpy()


def _to_numpy_array(array_like: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if cp is not None and isinstance(array_like, cp.ndarray):
        array_like = cp.asnumpy(array_like)
    return np.asarray(array_like, dtype=dtype)


def _collect_pixel_dataset(
    dataset: Dataset,
    num_classes: int,
    ignore_index: int,
    max_pixels_per_image: int,
    max_total_pixels: int,
    seed: int,
    tensor_device: Optional[torch.device] = None,
    use_gpu_backend: bool = False,
    augment_repeats: int = 0,
    augment_hflip_prob: float = 0.5,
    augment_vflip_prob: float = 0.5,
    augment_rot90: bool = True,
    augment_noise_std: float = 0.0,
) -> Tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    remaining = max_total_pixels if max_total_pixels > 0 else None
    repeat_count = 1 + max(0, int(augment_repeats))

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        for repeat_idx in range(repeat_count):
            if remaining is not None and remaining <= 0:
                break
            x_sel, y_sel = _sample_pixels_from_item(
                x,
                y,
                num_classes=num_classes,
                ignore_index=ignore_index,
                max_pixels_per_image=max_pixels_per_image,
                rng=rng,
                tensor_device=tensor_device,
                use_gpu_backend=use_gpu_backend,
                augment=repeat_idx > 0,
                augment_hflip_prob=augment_hflip_prob,
                augment_vflip_prob=augment_vflip_prob,
                augment_rot90=augment_rot90,
                augment_noise_std=augment_noise_std,
            )
            if x_sel.size == 0:
                continue

            if remaining is not None and y_sel.shape[0] > remaining:
                take = rng.choice(y_sel.shape[0], size=remaining, replace=False)
                x_sel = x_sel[take]
                y_sel = y_sel[take]
                remaining = 0
            elif remaining is not None:
                remaining -= y_sel.shape[0]

            xs.append(x_sel)
            ys.append(y_sel)

    if not xs:
        raise ValueError("No valid training pixels were collected. Check labels and ignore_index.")

    if use_gpu_backend:
        x_all = cp.concatenate(xs, axis=0)
        y_all = cp.concatenate(ys, axis=0)
    else:
        x_all = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y_all = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    return x_all, y_all


def _balance_pixel_dataset(
    x: Any,
    y: Any,
    num_classes: int,
    ignore_index: int,
    background_index: int,
    include_background: bool,
    max_multiplier: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_np = _to_numpy_array(x, dtype=np.float32)
    y_np = _to_numpy_array(y, dtype=np.int64)

    if x_np.ndim != 2:
        raise ValueError(f"Expected 2D pixel feature array, got shape {x_np.shape}")
    if y_np.ndim != 1:
        raise ValueError(f"Expected 1D pixel label array, got shape {y_np.shape}")
    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"Mismatched pixel features and labels: {x_np.shape[0]} vs {y_np.shape[0]}")

    rng = np.random.default_rng(seed)
    valid = (y_np != ignore_index) & (y_np >= 0) & (y_np < num_classes)
    if not np.any(valid):
        return x_np, y_np

    candidate_classes = [cls for cls in range(num_classes) if include_background or cls != background_index]
    class_counts = np.bincount(y_np[valid], minlength=num_classes)
    positive_counts = [int(class_counts[cls]) for cls in candidate_classes if class_counts[cls] > 0]
    if not positive_counts:
        return x_np, y_np

    majority = max(positive_counts)
    x_parts = [x_np]
    y_parts = [y_np]
    extra_pixels = 0

    for cls in candidate_classes:
        cls_idx = np.flatnonzero(y_np == cls)
        count = int(cls_idx.size)
        if count <= 0:
            continue
        multiplier = min(float(max_multiplier), float(np.sqrt(float(majority) / float(count))))
        target = int(np.ceil(count * max(1.0, multiplier)))
        if target <= count:
            continue
        add_count = target - count
        sampled_idx = rng.choice(cls_idx, size=add_count, replace=True)
        x_parts.append(x_np[sampled_idx])
        y_parts.append(y_np[sampled_idx])
        extra_pixels += add_count

    if extra_pixels <= 0:
        return x_np, y_np

    x_bal = np.concatenate(x_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(y_bal.shape[0])
    return x_bal[perm].astype(np.float32, copy=False), y_bal[perm].astype(np.int64, copy=False)


def _confusion_matrix_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    y_true = _to_numpy_array(y_true, dtype=np.int64).reshape(-1)
    y_pred = _to_numpy_array(y_pred, dtype=np.int64).reshape(-1)
    valid = (
        (y_true != ignore_index)
        & (y_true >= 0)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    idx = y_true[valid] * num_classes + y_pred[valid]
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes).astype(np.int64, copy=False)


def _metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    background_index: int,
    ignore_background: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape {cm.shape}")

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    valid_classes = [cls for cls in range(cm.shape[0]) if not (ignore_background and cls == background_index)]
    if valid_classes:
        macro_precision = float(np.mean(precision[valid_classes]))
        macro_recall = float(np.mean(recall[valid_classes]))
        macro_f1 = float(np.mean(f1[valid_classes]))
        support = cm.sum(axis=1)[valid_classes]
        support_sum = float(np.sum(support))
        weighted_f1 = float(np.sum(f1[valid_classes] * support) / support_sum) if support_sum > 0 else 0.0
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

    return precision, recall, f1, macro_precision, macro_recall, macro_f1, weighted_f1


def _per_class_iou_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    background_index: int,
    ignore_background: bool,
) -> np.ndarray:
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    per_class_iou = np.zeros(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        if ignore_background and cls == background_index:
            per_class_iou[cls] = np.nan
            continue
        inter = cm[cls, cls]
        union = cm[cls, :].sum() + cm[:, cls].sum() - inter
        if union > 0:
            per_class_iou[cls] = float(inter / union)
        else:
            per_class_iou[cls] = 0.0

    return per_class_iou


def _predict_image(model: Any, x: torch.Tensor, backend: str = "cpu") -> np.ndarray:
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    if x_np.ndim != 3:
        raise ValueError(f"Expected 3D feature tensor (C,H,W), got shape {x_np.shape}")
    c, h, w = x_np.shape

    if backend == "gpu":
        if cp is None:
            raise ImportError("cupy is required for GPU Random Forest prediction")
        x_gpu = cp.asarray(np.moveaxis(x_np, 0, -1).reshape(-1, c))
        pred = model.predict(x_gpu)
        if hasattr(cp, "asnumpy"):
            pred = cp.asnumpy(pred)
        else:
            pred = np.asarray(pred)
        return np.asarray(pred, dtype=np.int64).reshape(h, w)

    x_flat = np.moveaxis(x_np, 0, -1).reshape(-1, c)
    pred = model.predict(x_flat)
    return np.asarray(pred, dtype=np.int64).reshape(h, w)


def _evaluate_model(
    model: Any,
    dataset: Dataset,
    num_classes: int,
    ignore_index: int,
    background_index: int,
    ignore_background: bool,
    backend: str = "cpu",
) -> Dict[str, Any]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        pred = _predict_image(model, x, backend=backend)
        y_np = y.detach().cpu().numpy().astype(np.int64, copy=False)
        confusion += _confusion_matrix_from_predictions(y_np, pred, num_classes, ignore_index)

    total = int(confusion.sum())
    pixel_acc = float(np.trace(confusion)) / float(total) if total > 0 else 0.0

    ious = _per_class_iou_from_confusion_matrix(confusion, background_index, ignore_background)
    valid_classes = [cls for cls in range(num_classes) if not (ignore_background and cls == background_index)]
    miou = float(np.mean(ious[valid_classes])) if valid_classes else 0.0

    precision, recall, f1, macro_precision, macro_recall, macro_f1, weighted_f1 = _metrics_from_confusion_matrix(
        confusion,
        background_index=background_index,
        ignore_background=ignore_background,
    )

    return {
        "confusion_matrix": confusion,
        "pixel_acc": pixel_acc,
        "miou": miou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_iou": ious,
    }


def _save_val_visualizations(
    model: Any,
    val_ds: Dataset,
    output_dir: Path,
    num_classes: int,
    ignore_index: int,
    max_images: int,
    backend: str = "cpu",
) -> None:
    if val_ds is None or len(val_ds) == 0 or max_images <= 0:
        return

    vis_dir = output_dir / "val_visualizations"
    _ensure_dir(vis_dir)

    n = min(max_images, len(val_ds))
    for i in range(n):
        x, y = val_ds[i]
        pred = _predict_image(model, x, backend=backend)

        x_np = x.detach().cpu().numpy()
        rgb = x_np[:3] if x_np.shape[0] >= 3 else np.repeat(x_np[:1], 3, axis=0)
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        y_np = y.detach().cpu().numpy().astype(np.int64, copy=False)
        y_rgb = labels_to_rgb(y_np, num_classes=num_classes, ignore_index=ignore_index)
        p_rgb = labels_to_rgb(pred, num_classes=num_classes, ignore_index=ignore_index)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Input (first 3 channels)")
        axes[0].axis("off")

        axes[1].imshow(y_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(p_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(vis_dir / f"val_sample_{i:02d}.png", dpi=150)
        plt.close(fig)


def _save_metrics_csv(path: Path, metrics: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list, tuple, dict)):
                continue
            writer.writerow([key, value])


def _save_feature_importance(path: Path, importances: np.ndarray) -> None:
    _ensure_dir(path.parent)
    np.save(path.with_suffix(".npy"), importances.astype(np.float32, copy=False))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "importance"])
        for idx, value in enumerate(importances):
            writer.writerow([idx, f"{float(value):.8f}"])


def _run_train_command(args: argparse.Namespace) -> Dict[str, float]:
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    merged_cfg = _materialize_run_config(cfg, args)
    train_args = _namespace_from_config(merged_cfg)
    return train(train_args)


def train(args: argparse.Namespace) -> Dict[str, float]:
    start_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    backend = str(getattr(args, "rf_backend", "auto")).strip().lower()
    if backend == "auto":
        backend = "gpu" if _gpu_rf_available() else "cpu"
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported rf backend: {backend}")
    if backend == "gpu" and not _gpu_rf_available():
        raise ImportError(
            "GPU RF backend requested, but cupy/cuml or CUDA are not available in this environment."
        )
    tensor_device = torch.device(args.device if getattr(args, "device", "") else ("cuda:0" if backend == "gpu" else "cpu"))
    if backend == "gpu" and tensor_device.type != "cuda":
        raise ValueError("GPU backend requires a CUDA device (e.g. cuda:0) in runtime.device.")

    train_ds, val_ds, train_for_sampling = _build_train_val_datasets(args)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train or validation dataset is empty.")

    sample_x, sample_y = train_ds[0]
    in_channels = int(sample_x.shape[0])
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Sample feature shape (C,H,W)=({sample_x.shape[0]},{sample_x.shape[1]},{sample_x.shape[2]})")

    valid_ratio, y_min, y_max = estimate_label_valid_ratio(
        train_for_sampling,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_samples=min(16, len(train_for_sampling)),
    )
    if valid_ratio < 0.95:
        sample_unique = torch.unique(sample_y.reshape(-1)).cpu().numpy().tolist()
        raise ValueError(
            "Label sanity check failed: most label pixels are outside valid class range. "
            f"valid_ratio={valid_ratio:.4f}, min={y_min}, max={y_max}, "
            f"num_classes={args.num_classes}, ignore_index={args.ignore_index}, "
            f"sample_unique_head={sample_unique[:12]}"
        )

    x_train, y_train = _collect_pixel_dataset(
        train_ds,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_pixels_per_image=args.max_train_pixels_per_image,
        max_total_pixels=args.max_train_pixels_total,
        seed=args.seed,
        tensor_device=None,
        use_gpu_backend=False,
        augment_repeats=args.augment_repeats,
        augment_hflip_prob=args.augment_hflip_prob,
        augment_vflip_prob=args.augment_vflip_prob,
        augment_rot90=bool(args.augment_rot90),
        augment_noise_std=args.augment_noise_std,
    )
    print(f"Collected training pixels: {x_train.shape[0]} x {x_train.shape[1]}")

    if bool(getattr(args, "balance_minority_pixels", False)):
        x_train, y_train = _balance_pixel_dataset(
            x_train,
            y_train,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            background_index=args.background_index,
            include_background=bool(getattr(args, "balance_include_background", False)),
            max_multiplier=float(getattr(args, "balance_max_multiplier", 4.0)),
            seed=args.seed,
        )
        print(f"Balanced training pixels: {x_train.shape[0]} x {x_train.shape[1]}")

    if backend == "gpu":
        x_train = cp.asarray(x_train)
        y_train = cp.asarray(y_train)

    if args.max_features_mode == "input_channels":
        max_features = in_channels
    else:
        max_features = args.max_features_value

    rf_kwargs = {
        "n_estimators": getattr(args, "rf_n_estimators", RF_N_ESTIMATORS),
        "max_features": max_features,
        "min_samples_leaf": getattr(args, "rf_min_samples_leaf", RF_MIN_SAMPLES_LEAF),
        "bootstrap": getattr(args, "rf_bootstrap", RF_BOOTSTRAP),
        "max_samples": getattr(args, "rf_max_samples", RF_MAX_SAMPLES),
        "random_state": getattr(args, "rf_random_state", RF_RANDOM_STATE),
    }
    if backend == "cpu":
        rf_kwargs["n_jobs"] = args.n_jobs
        rf_kwargs["verbose"] = args.verbose

    rf_cls = CuMLRandomForestClassifier if backend == "gpu" else RandomForestClassifier
    model = rf_cls(**rf_kwargs)

    print(
        "Random Forest config: "
        f"backend={backend}, "
        f"n_estimators={getattr(args, 'rf_n_estimators', RF_N_ESTIMATORS)}, max_features={max_features}, "
        f"min_samples_leaf={getattr(args, 'rf_min_samples_leaf', RF_MIN_SAMPLES_LEAF)}, "
        f"bootstrap={getattr(args, 'rf_bootstrap', RF_BOOTSTRAP)}, "
        f"bag_fraction={getattr(args, 'rf_max_samples', RF_MAX_SAMPLES)}, "
        f"random_state={getattr(args, 'rf_random_state', RF_RANDOM_STATE)}"
    )

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    if backend == "gpu" and cp is not None and hasattr(cp, "asnumpy"):
        train_pred = cp.asnumpy(train_pred)
    train_confusion = _confusion_matrix_from_predictions(y_train, train_pred, args.num_classes, args.ignore_index)
    train_total = int(train_confusion.sum())
    train_pixel_acc = float(np.trace(train_confusion)) / float(train_total) if train_total > 0 else 0.0
    train_ious = _per_class_iou_from_confusion_matrix(
        train_confusion,
        background_index=args.background_index,
        ignore_background=bool(args.ignore_background_in_metrics),
    )
    valid_classes = [
        cls
        for cls in range(args.num_classes)
        if not (bool(args.ignore_background_in_metrics) and cls == args.background_index)
    ]
    train_miou = float(np.mean(train_ious[valid_classes])) if valid_classes else 0.0

    val_stats = _evaluate_model(
        model=model,
        dataset=val_ds,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
        ignore_background=bool(args.ignore_background_in_metrics),
        backend=backend,
    )

    print(
        f"Train pixel_acc: {train_pixel_acc:.4f}, train_mIoU: {train_miou:.4f}, "
        f"val_pixel_acc: {val_stats['pixel_acc']:.4f}, val_mIoU: {val_stats['miou']:.4f}, "
        f"val_macro_F1: {val_stats['macro_f1']:.4f}, val_weighted_F1: {val_stats['weighted_f1']:.4f}"
    )

    model_path = out_dir / "random_forest_segmentation.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved Random Forest model to {model_path}")

    save_confusion_matrix_visualization(
        val_stats["confusion_matrix"],
        output_dir=out_dir,
        prefix="val",
        class_names=[str(i) for i in range(args.num_classes)],
        normalize=True,
    )
    dump_per_class_iou_report(
        val_stats["per_class_iou"],
        output_dir=out_dir,
        prefix="best_val",
        per_class_f1=val_stats["f1"],
        weighted_f1=float(val_stats["weighted_f1"]),
    )

    if args.save_val_samples > 0 and len(val_ds) > 0 and plt is not None:
        _save_val_visualizations(
            model=model,
            val_ds=val_ds,
            output_dir=out_dir,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_images=args.save_val_samples,
            backend=backend,
        )
        print(f"Saved validation visualizations to {out_dir / 'val_visualizations'}")
    elif args.save_val_samples > 0 and plt is None:
        print("matplotlib is not available; skipping validation visualizations.")

    if getattr(model, "feature_importances_", None) is not None:
        _save_feature_importance(out_dir / "random_forest_feature_importances.csv", model.feature_importances_)

    metrics = {
        "train_pixel_acc": float(train_pixel_acc),
        "train_miou": float(train_miou),
        "val_pixel_acc": float(val_stats["pixel_acc"]),
        "val_miou": float(val_stats["miou"]),
        "val_macro_precision": float(val_stats["macro_precision"]),
        "val_macro_recall": float(val_stats["macro_recall"]),
        "val_macro_f1": float(val_stats["macro_f1"]),
        "val_weighted_f1": float(val_stats["weighted_f1"]),
        "elapsed_seconds": float(time.time() - start_time),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    _save_metrics_csv(out_dir / "metrics.csv", metrics)

    print(f"Saved metrics to {out_dir / 'metrics.json'}")
    print(f"Saved metrics to {out_dir / 'metrics.csv'}")

    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Random Forest segmentation model from embedding or raw pixel features."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run training from a YAML/JSON config")
    p_train.add_argument("--config", type=str, required=True, help="YAML/JSON config path")
    p_train.add_argument("--output_dir", type=str, default="", help="Override runtime.output_dir")
    p_train.add_argument("--device", type=str, default="", help="Override runtime.device")

    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        _run_train_command(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()