import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, List, Optional
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
import yaml


# ---------- Data preparation helpers ----------


def prepare_features_from_embeddings(emb_npy: Path, embedding_key: str = "auto") -> np.ndarray:
    """Load embeddings from .npy and return features as (C,H,W) float32.

    Supports two formats:
      - embeddings: (H,W,64)
      - embeddings_per_time: (T,H,W,64) -> (T*64,H,W)
    """
    data = np.load(emb_npy, allow_pickle=True)

    # np.save(dict) -> .npy is loaded as 0-d object array; unwrap to mapping-like object.
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()

    # Support plain .npy embeddings saved as either arrays or dict-like object arrays.
    if isinstance(data, np.ndarray):
        e = data
        if e.ndim == 4:
            # (T,H,W,64) or (T,64,H,W)
            if e.shape[-1] == 64:
                e_chw = np.transpose(e, (0, 3, 1, 2))
            elif e.shape[1] == 64:
                e_chw = e
            else:
                raise ValueError(f"Unexpected ndarray embedding shape {e.shape} in {emb_npy}")
            T, C, H, W = e_chw.shape
            feats = e_chw.reshape(T * C, H, W)
        elif e.ndim == 3:
            # (H,W,64) or already (C,H,W)
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
                T, C, H, W = e_chw.shape
                feats = e_chw.reshape(T * C, H, W)
            elif e.ndim == 3:
                if e.shape[-1] != 64:
                    raise ValueError(f"Unexpected {embedding_key} shape {e.shape} in {emb_npy}")
                feats = np.transpose(e, (2, 0, 1))
            else:
                raise ValueError(f"Unsupported {embedding_key} ndim={e.ndim} in {emb_npy}")
        elif "embeddings_per_time" in data:
            e = data["embeddings_per_time"]  # (T,H,W,64)
            if e.ndim != 4 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings_per_time shape {e.shape} in {emb_npy}")
            e_chw = np.transpose(e, (0, 3, 1, 2))  # (T,64,H,W)
            T, C, H, W = e_chw.shape
            feats = e_chw.reshape(T * C, H, W)
        elif "embeddings" in data:
            e = data["embeddings"]  # (H,W,64)
            if e.ndim != 3 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings shape {e.shape} in {emb_npy}")
            feats = np.transpose(e, (2, 0, 1))  # (64,H,W)
        elif "embeddings_native" in data:
            e = data["embeddings_native"]  # typically (H,W,64), e.g. 64x64 native map
            if e.ndim != 3 or e.shape[-1] != 64:
                raise ValueError(f"Unexpected embeddings_native shape {e.shape} in {emb_npy}")
            feats = np.transpose(e, (2, 0, 1))  # (64,H,W)
        else:
            raise ValueError(
                f"None of 'embeddings_per_time'/'embeddings'/'embeddings_native' found in {emb_npy}"
            )
    elif embedding_key != "auto":
        if embedding_key not in data:
            raise ValueError(f"Embedding key '{embedding_key}' not found in {emb_npy}")
        e = data[embedding_key]
        if e.ndim == 4:
            if e.shape[-1] != 64:
                raise ValueError(f"Unexpected {embedding_key} shape {e.shape} in {emb_npy}")
            e_chw = np.transpose(e, (0, 3, 1, 2))
            T, C, H, W = e_chw.shape
            feats = e_chw.reshape(T * C, H, W)
        elif e.ndim == 3:
            if e.shape[-1] != 64:
                raise ValueError(f"Unexpected {embedding_key} shape {e.shape} in {emb_npy}")
            feats = np.transpose(e, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported {embedding_key} ndim={e.ndim} in {emb_npy}")
    elif "embeddings_per_time" in data:
        e = data["embeddings_per_time"]  # (T,H,W,64)
        if e.ndim != 4 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings_per_time shape {e.shape} in {emb_npy}")
        e_chw = np.transpose(e, (0, 3, 1, 2))  # (T,64,H,W)
        T, C, H, W = e_chw.shape
        feats = e_chw.reshape(T * C, H, W)
    elif "embeddings" in data:
        e = data["embeddings"]  # (H,W,64)
        if e.ndim != 3 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings shape {e.shape} in {emb_npy}")
        feats = np.transpose(e, (2, 0, 1))  # (64,H,W)
    elif "embeddings_native" in data:
        e = data["embeddings_native"]  # typically (H,W,64), e.g. 64x64 native map
        if e.ndim != 3 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings_native shape {e.shape} in {emb_npy}")
        feats = np.transpose(e, (2, 0, 1))  # (64,H,W)
    else:
        raise ValueError(
            f"None of 'embeddings_per_time'/'embeddings'/'embeddings_native' found in {emb_npy}"
        )

    # Robustness: some AEF tiles may contain sparse NaNs after reprojection.
    # Replace non-finite values so normalization/loss computation stays stable.
    feats = feats.astype(np.float32, copy=False)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def resize_labels_to(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Resize label mask (H_l,W_l) to match features (C,H_f,W_f) using nearest neighbor."""
    C, H_f, W_f = features.shape
    H_l, W_l = labels.shape

    if (H_f, W_f) == (H_l, W_l):
        return labels

    # Use interpolate to guarantee exact output size.
    t = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = nn.functional.interpolate(t, size=(H_f, W_f), mode="nearest")
    return t.squeeze(0).squeeze(0).numpy().astype(labels.dtype)


def load_label_mask_from_file(label_file: Path, label_key: str = "labels") -> np.ndarray:
    """Load a 2D label mask from .npy/.npz files.

    Supports plain 2D arrays as well as object payloads that wrap a dict with a
    ``labels`` entry, which is the format used by the PASTIS-R per-patch files.
    """
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
                    raise KeyError(
                        f"Label key '{label_key}' not found in {label_file}. Available keys: {available}"
                    )
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


def resize_features_to(features: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize features (C,H,W) to exact target size using bilinear interpolation."""
    _, h, w = features.shape
    if (h, w) == (out_h, out_w):
        return features
    t = torch.from_numpy(features).unsqueeze(0)
    t = nn.functional.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)


class EmbeddingSegmentationDataset(Dataset):
    """Segmentation dataset from AEF embeddings and integer labels.

    Supports two modes:
      - Single-sample: embeddings_path is a file (one .npy) -> len=1.
      - Multi-sample: embeddings_path is a directory -> all *.npy inside
        are treated as separate samples.

    In both cases, labels provide a (H,W) label mask which is resized
    to each sample's spatial resolution as needed.

    Optionally, when per_patch_labels=True, labels_path is treated as a
    directory containing per-patch label files whose names follow the
    pattern ParcelIDs_XXXXX.npy, and embeddings are named
    embedding_XXXXX*.npy (or any name ending with the same numeric XXXXX).
    """
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
        # 可选：用于“按索引一一对应”的标签模式（例如 embedding_0000.npy ↔ 20250413_000_label.npy）。
        self.index_label_files: Optional[List[Path]] = None

        if per_patch_labels:
            if not labels_path.is_dir():
                raise ValueError(
                    f"With per_patch_labels=True, labels_path must be a directory, got {labels_path}"
                )
            self.labels_dir = labels_path
            self.labels_np = None
        else:
            if labels_path.suffix.lower() != ".npy":
                raise ValueError(f"Labels must be a .npy file, got {labels_path}")

            # Load a single global label mask once; will be resized per-sample if needed.
            labels = np.load(labels_path, allow_pickle=True)

            if labels.ndim != 2:
                raise ValueError(f"Expected labels with shape (H,W), got {labels.shape}")
            self.labels_np = labels
            self.labels_dir = None

        if embeddings_path.is_dir():
            # Collect only .npy embedding files.
            self.files = sorted([p for p in embeddings_path.glob("*.npy") if p.is_file()])
            if not self.files:
                raise FileNotFoundError(
                    f"No embedding files (.npy) found in embeddings directory {embeddings_path}"
                )
        else:
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings npy not found: {embeddings_path}")
            if embeddings_path.suffix.lower() != ".npy":
                raise ValueError(f"Embeddings must be a .npy file, got {embeddings_path}")
            self.files = [embeddings_path]

        # 若是 per-patch 模式，但标签文件名并非 sample_*/ParcelIDs_*，
        # 且标签数量与 embedding 数量一致，则启用“按索引一一对应”的匹配策略。
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
        feats = prepare_features_from_embeddings(emb_file, embedding_key=self.embedding_key)  # (C,H,W)
        if self.resample_size > 0:
            feats = resize_features_to(feats, self.resample_size, self.resample_size)

        if self.per_patch_labels:
            label_file: Optional[Path] = None

            # 优先：若启用了按索引对应模式，则直接使用预先排序好的标签列表。
            if self.index_label_files is not None:
                if idx >= len(self.index_label_files):
                    raise IndexError(
                        f"Index {idx} out of range for index-based labels list of length {len(self.index_label_files)}"
                    )
                label_file = self.index_label_files[idx]
            else:
                # 默认 Per-patch 模式：根据 embedding 文件名推断对应的标签文件。
                # 兼容主要组织方式：
                #   1) 原始示例：embedding_XXXXX*.npy  <-> ParcelIDs_XXXXX_labels.npy
                #   2) 本项目当前数据：embedding_XXXXXX_YY.npy <-> sample_XXXXXX_YY_label.npy
                #   3) MTS12 AEF：eopath_ID_COL_ROW.npy <-> ParcelIDs_ID.npy
                #   4) PASTIS-R 常见命名：embedding_XXXXX.npy <-> ParcelIDs_XXXXX.npy
                stem = emb_file.stem  # e.g., "embedding_00241_00"

                # 先去掉前缀 "embedding_"，保留完整 patch id（如 "00241_00"、"126582_37" 等）。
                patch_token = stem
                if patch_token.startswith("embedding_"):
                    patch_token = patch_token[len("embedding_") :]

                # 候选标签文件名按常见模式依次尝试。
                candidate_paths: List[Path] = []
                if self.labels_dir is not None:
                    # 当前数据集使用的 sample_XXXX_YY_label.npy
                    candidate_paths.append(self.labels_dir / f"sample_{patch_token}_label.npy")

                    # MTS12 AEF 命名：eopath_<id>_<col>_<row>，标签按 ID 命名。
                    m_eopath = re.match(r"^eopath_(\d+)_\d+_\d+$", patch_token)
                    if m_eopath is not None:
                        patch_id = int(m_eopath.group(1))
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id:05d}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id}_labels.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id:05d}.npy")
                        candidate_paths.append(self.labels_dir / f"ParcelIDs_{patch_id}.npy")

                    # 向后兼容原始 ParcelIDs_XXXXX_labels.npy / ParcelIDs_XXXXX.npy 方案（仅使用末尾数字串）。
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
                    f"{emb_file.name}; tried: "
                    + ", ".join(str(c) for c in candidate_paths)
                )

            labels_np = load_label_mask_from_file(label_file)
        else:
            labels_np = self.labels_np  # type: ignore[assignment]


        labels_np = labels_np.astype(np.int64)

        labels_resized = resize_labels_to(feats, labels_np)
        features = torch.from_numpy(feats)  # (C,H,W)
        labels = torch.from_numpy(labels_resized.astype(np.int64))  # (H,W)
        return features, labels


class AugmentedTrainDataset(Dataset):
    """Wrap a dataset with stochastic train-time augmentation and repeat factor.

    This creates a virtual larger dataset via repeated indexing while applying
    random geometric/intensity transforms to embeddings each access.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        repeat_factor: int = 1,
        flip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        noise_std: float = 0.01,
        gain_std: float = 0.05,
        crop_size: int = 0,
        foreground_crop_prob: float = 0.0,
        background_index: int = 0,
        ignore_index: int = 19,
        sample_foreground_ratios: Optional[np.ndarray] = None,
        fg_ratio_threshold: float = 0.2,
        fg_aug_boost: float = 1.5,
    ):
        self.base_dataset = base_dataset
        self.repeat_factor = max(1, int(repeat_factor))
        self.flip_prob = float(np.clip(flip_prob, 0.0, 1.0))
        self.rot90_prob = float(np.clip(rot90_prob, 0.0, 1.0))
        self.noise_std = max(0.0, float(noise_std))
        self.gain_std = max(0.0, float(gain_std))
        self.crop_size = max(0, int(crop_size))
        self.foreground_crop_prob = float(np.clip(foreground_crop_prob, 0.0, 1.0))
        self.background_index = int(background_index)
        self.ignore_index = int(ignore_index)
        self.fg_ratio_threshold = float(np.clip(fg_ratio_threshold, 0.0, 1.0))
        self.fg_aug_boost = max(1.0, float(fg_aug_boost))
        self.sample_foreground_ratios = None
        if sample_foreground_ratios is not None:
            ratios = np.asarray(sample_foreground_ratios, dtype=np.float32).reshape(-1)
            if len(ratios) != len(base_dataset):
                raise ValueError(
                    f"sample_foreground_ratios length {len(ratios)} does not match base dataset length {len(base_dataset)}"
                )
            self.sample_foreground_ratios = ratios

    def __len__(self) -> int:
        return len(self.base_dataset) * self.repeat_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx % len(self.base_dataset)
        x, y = self.base_dataset[base_idx]

        fg_ratio = None
        if self.sample_foreground_ratios is not None:
            fg_ratio = float(self.sample_foreground_ratios[base_idx])

        # Clone to avoid in-place mutation of tensors returned by base dataset.
        x = x.clone()
        y = y.clone()

        adaptive_flip_prob = self.flip_prob
        adaptive_rot90_prob = self.rot90_prob
        adaptive_noise_std = self.noise_std
        adaptive_gain_std = self.gain_std
        adaptive_crop_size = self.crop_size
        adaptive_foreground_crop_prob = self.foreground_crop_prob

        if fg_ratio is not None and fg_ratio >= self.fg_ratio_threshold:
            span = max(1e-6, 1.0 - self.fg_ratio_threshold)
            strength = float(np.clip((fg_ratio - self.fg_ratio_threshold) / span, 0.0, 1.0))
            boost = 1.0 + (self.fg_aug_boost - 1.0) * strength
            adaptive_flip_prob = float(np.clip(self.flip_prob * (0.8 + 0.2 * boost), 0.0, 1.0))
            adaptive_rot90_prob = float(np.clip(self.rot90_prob * (0.8 + 0.2 * boost), 0.0, 1.0))
            adaptive_noise_std = self.noise_std * boost
            adaptive_gain_std = self.gain_std * boost
            adaptive_foreground_crop_prob = float(np.clip(self.foreground_crop_prob * boost, 0.0, 1.0))
            if self.crop_size > 0:
                adaptive_crop_size = min(
                    max(1, int(round(self.crop_size * (1.0 + 0.25 * strength)))),
                    x.shape[-2],
                    x.shape[-1],
                )

        # Random flips (geometric transforms must be applied to both x and y).
        if torch.rand(1).item() < adaptive_flip_prob:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        if torch.rand(1).item() < adaptive_flip_prob:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])

        # Random 90-degree rotation avoids interpolation artifacts for labels.
        if torch.rand(1).item() < adaptive_rot90_prob:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k=k, dims=[1, 2])
                y = torch.rot90(y, k=k, dims=[0, 1])

        # Intensity jitter for embeddings (labels unchanged).
        if adaptive_gain_std > 0:
            gain = 1.0 + torch.randn(1, dtype=x.dtype).item() * adaptive_gain_std
            x = x * gain
        if adaptive_noise_std > 0:
            x = x + torch.randn_like(x) * adaptive_noise_std

        # Foreground-aware random crop increases effective fg pixel ratio.
        if adaptive_crop_size > 0:
            _, h, w = x.shape
            cs = min(adaptive_crop_size, h, w)
            if cs < h or cs < w:
                top, left = 0, 0
                use_fg_center = torch.rand(1).item() < adaptive_foreground_crop_prob
                if use_fg_center:
                    fg_mask = (y != self.background_index) & (y != self.ignore_index)
                    fg_idx = torch.nonzero(fg_mask, as_tuple=False)
                    if fg_idx.numel() > 0:
                        pick = fg_idx[torch.randint(0, fg_idx.shape[0], (1,)).item()]
                        cy, cx = int(pick[0].item()), int(pick[1].item())
                        top = max(0, min(h - cs, cy - cs // 2))
                        left = max(0, min(w - cs, cx - cs // 2))
                    else:
                        top = int(torch.randint(0, h - cs + 1, (1,)).item())
                        left = int(torch.randint(0, w - cs + 1, (1,)).item())
                else:
                    top = int(torch.randint(0, h - cs + 1, (1,)).item())
                    left = int(torch.randint(0, w - cs + 1, (1,)).item())
                x = x[:, top:top + cs, left:left + cs]
                y = y[top:top + cs, left:left + cs]

        return x, y


class FeatureNormalizeDataset(Dataset):
    """Apply fixed channel-wise normalization to features in a dataset."""

    def __init__(self, base_dataset: Dataset, mean: torch.Tensor, std: torch.Tensor):
        self.base_dataset = base_dataset
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base_dataset[idx]
        x = (x - self.mean) / self.std
        return x, y


def estimate_feature_channel_stats(dataset: Dataset, max_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate channel-wise mean/std on training features only.

    max_samples=0 means use all samples.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot estimate feature stats from empty dataset")

    if max_samples > 0:
        n = min(n, int(max_samples))

    sum_c = None
    sumsq_c = None
    count = 0

    for i in range(n):
        x, _ = dataset[i]  # (C,H,W)
        x = x.float()
        c = x.shape[0]
        flat = x.reshape(c, -1)
        s = flat.sum(dim=1)
        ss = (flat * flat).sum(dim=1)
        if sum_c is None:
            sum_c = s
            sumsq_c = ss
        else:
            sum_c = sum_c + s
            sumsq_c = sumsq_c + ss
        count += flat.shape[1]

    assert sum_c is not None and sumsq_c is not None
    mean = sum_c / max(1, count)
    var = (sumsq_c / max(1, count)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


def estimate_sample_foreground_ratios(
    dataset: Dataset,
    background_index: int,
    ignore_index: int,
) -> np.ndarray:
    """Estimate per-sample foreground ratio for curriculum warmup."""
    ratios = []
    for i in range(len(dataset)):
        _x, y = dataset[i]
        valid = y != ignore_index
        denom = int(valid.sum().item())
        if denom <= 0:
            ratios.append(0.0)
            continue
        fg = ((y != background_index) & valid).sum().item()
        ratios.append(float(fg) / float(denom))
    return np.asarray(ratios, dtype=np.float32)


def estimate_label_valid_ratio(
    dataset: Dataset,
    num_classes: int,
    ignore_index: int,
    max_samples: int = 16,
) -> tuple[float, int, int]:
    """Estimate how many label pixels are in valid range.

    A valid pixel is either ignore_index or in [0, num_classes-1].
    """
    n = min(len(dataset), max(1, int(max_samples)))
    if n <= 0:
        return 1.0, 0, 0

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


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance"):
        super().__init__()
        if norm == "batch":
            norm1 = nn.BatchNorm2d(out_ch)
            norm2 = nn.BatchNorm2d(out_ch)
        elif norm == "instance":
            norm1 = nn.InstanceNorm2d(out_ch, affine=True)
            norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            # GroupNorm is more stable than BatchNorm for small/imbalanced batches.
            g = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
            norm1 = nn.GroupNorm(g, out_ch)
            norm2 = nn.GroupNorm(g, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    g = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
    return nn.GroupNorm(g, channels)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(norm, out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = torch.relu(x + residual)
        x = self.se(x)
        return self.dropout(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        theta_skip = self.theta(skip)
        phi_gate = self.phi(gate)

        # 对齐空间尺寸，避免由于下采样/上采样舍入导致的 1 像素差异
        if theta_skip.shape[-2:] != phi_gate.shape[-2:]:
            h = min(theta_skip.shape[-2], phi_gate.shape[-2])
            w = min(theta_skip.shape[-1], phi_gate.shape[-1])

            def center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
                _, _, th, tw = t.shape
                start_h = max((th - target_h) // 2, 0)
                start_w = max((tw - target_w) // 2, 0)
                return t[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

            theta_skip = center_crop(theta_skip, h, w)
            phi_gate = center_crop(phi_gate, h, w)

            # 同时裁剪 skip 到相同空间尺寸
            skip = center_crop(skip, h, w)

        alpha = torch.sigmoid(self.psi(torch.relu(theta_skip + phi_gate)))
        return skip * alpha


class UNet(nn.Module):
    """2-D U-Net with a VGG16-style encoder and InstanceNorm blocks."""

    def __init__(self, in_channels: int, num_classes: int = 20, base_ch: int = 32, norm: str = "instance"):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch, norm=norm)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2, norm=norm)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4, norm=norm)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8, norm=norm)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16, norm=norm)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8, norm=norm)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, norm=norm)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    @staticmethod
    def _match_spatial(skip: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if skip.shape[-2:] == gate.shape[-2:]:
            return skip, gate

        h = min(skip.shape[-2], gate.shape[-2])
        w = min(skip.shape[-1], gate.shape[-1])

        def center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            _, _, th, tw = t.shape
            start_h = max((th - target_h) // 2, 0)
            start_w = max((tw - target_w) // 2, 0)
            return t[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

        return center_crop(skip, h, w), center_crop(gate, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        xb = self.bottleneck(self.pool4(x4))

        # Decoder with skip connections
        x = self.up4(xb)
        x4, x = self._match_spatial(x4, x)
        x = torch.cat([x4, x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x3, x = self._match_spatial(x3, x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x2, x = self._match_spatial(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x1, x = self._match_spatial(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)


class UNetDeep(nn.Module):
    """Deeper U-Net with one additional down/up stage.

    Channels become:
      base, 2*base, 4*base, 8*base, 16*base, bottleneck=32*base
    """

    def __init__(self, in_channels: int, num_classes: int = 20, base_ch: int = 32, norm: str = "instance"):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch, norm=norm)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2, norm=norm)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4, norm=norm)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8, norm=norm)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = DoubleConv(base_ch * 8, base_ch * 16, norm=norm)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 16, base_ch * 32, norm=norm)

        self.up5 = nn.ConvTranspose2d(base_ch * 32, base_ch * 16, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(base_ch * 32, base_ch * 16, norm=norm)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8, norm=norm)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, norm=norm)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    @staticmethod
    def _match_spatial(skip: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if skip.shape[-2:] == gate.shape[-2:]:
            return skip, gate

        h = min(skip.shape[-2], gate.shape[-2])
        w = min(skip.shape[-1], gate.shape[-1])

        def center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            _, _, th, tw = t.shape
            start_h = max((th - target_h) // 2, 0)
            start_w = max((tw - target_w) // 2, 0)
            return t[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

        return center_crop(skip, h, w), center_crop(gate, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))
        xb = self.bottleneck(self.pool5(x5))

        x = self.up5(xb)
        x5, x = self._match_spatial(x5, x)
        x = torch.cat([x5, x], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x4, x = self._match_spatial(x4, x)
        x = torch.cat([x4, x], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x3, x = self._match_spatial(x3, x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x2, x = self._match_spatial(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x1, x = self._match_spatial(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)


class UNetResSE(nn.Module):
    """UNet variant with residual+SE blocks and attention-gated skip connections."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 20,
        base_ch: int = 32,
        norm: str = "group",
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth not in (4, 5):
            raise ValueError(f"UNetResSE supports depth 4 or 5, got {depth}")

        enc_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        if depth == 5:
            enc_channels.append(base_ch * 16)

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for c in enc_channels:
            self.enc_blocks.append(ResidualSEBlock(in_ch, c, norm=norm, dropout=dropout * 0.5))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = c

        bottleneck_ch = enc_channels[-1] * 2
        self.bottleneck = ResidualSEBlock(enc_channels[-1], bottleneck_ch, norm=norm, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        cur_ch = bottleneck_ch
        for skip_ch in reversed(enc_channels):
            self.upconvs.append(nn.ConvTranspose2d(cur_ch, skip_ch, kernel_size=2, stride=2))
            inter_ch = max(skip_ch // 2, 8)
            self.attn_gates.append(AttentionGate(skip_ch=skip_ch, gate_ch=skip_ch, inter_ch=inter_ch))
            self.dec_blocks.append(ResidualSEBlock(skip_ch * 2, skip_ch, norm=norm, dropout=dropout * 0.5))
            cur_ch = skip_ch

        self.out_conv = nn.Conv2d(cur_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, attn, dec, skip in zip(self.upconvs, self.attn_gates, self.dec_blocks, reversed(skips)):
            x = up(x)
            skip = attn(skip, x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.out_conv(x)


class SeparableConvBlock(nn.Module):
    """Depthwise-separable conv block for lightweight segmentation backbones."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = _make_norm(norm, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class LightweightConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        self.block = nn.Sequential(
            SeparableConvBlock(in_ch, out_ch, norm=norm),
            SeparableConvBlock(out_ch, out_ch, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetLightweight(nn.Module):
    """Lightweight UNet using depthwise-separable blocks and bilinear upsampling."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 20,
        base_ch: int = 24,
        norm: str = "group",
        depth: int = 4,
    ):
        super().__init__()
        if depth not in (4, 5):
            raise ValueError(f"UNetLightweight supports depth 4 or 5, got {depth}")

        enc_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        if depth == 5:
            enc_channels.append(base_ch * 16)

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for c in enc_channels:
            self.enc_blocks.append(LightweightConv(in_ch, c, norm=norm))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = c

        bottleneck_ch = enc_channels[-1] * 2
        self.bottleneck = LightweightConv(enc_channels[-1], bottleneck_ch, norm=norm)

        self.up_projs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        cur_ch = bottleneck_ch
        for skip_ch in reversed(enc_channels):
            self.up_projs.append(nn.Conv2d(cur_ch, skip_ch, kernel_size=1, bias=False))
            self.dec_blocks.append(LightweightConv(skip_ch * 2, skip_ch, norm=norm))
            cur_ch = skip_ch

        self.out_conv = nn.Conv2d(cur_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for proj, dec, skip in zip(self.up_projs, self.dec_blocks, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = proj(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.out_conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group"):
        super().__init__()
        rates = [1, 6, 12, 18]
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                        _make_norm(norm, out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                        _make_norm(norm, out_ch),
                        nn.ReLU(inplace=True),
                    )
                )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, kernel_size=1, bias=False),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = nn.functional.interpolate(img, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(img)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabLite(nn.Module):

    def __init__(self, in_channels: int, num_classes: int = 20, base_ch: int = 32, norm: str = "group"):
        super().__init__()
        self.stem = ResidualSEBlock(in_channels, base_ch, norm=norm, dropout=0.0)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualSEBlock(base_ch, base_ch * 2, norm=norm, dropout=0.05))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualSEBlock(base_ch * 2, base_ch * 4, norm=norm, dropout=0.05))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualSEBlock(base_ch * 4, base_ch * 8, norm=norm, dropout=0.1))

        self.aspp = ASPP(base_ch * 8, base_ch * 4, norm=norm)

        self.low_proj = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=1, bias=False),
            _make_norm(norm, base_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_ch * 5, base_ch * 2, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(base_ch * 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        low = self.stem(x)                 # H,W
        x = self.down1(low)                # H/2,W/2
        x = self.down2(x)                  # H/4,W/4
        x = self.down3(x)                  # H/8,W/8
        x = self.aspp(x)                   # H/8,W/8

        x = nn.functional.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.fuse(x)
        x = self.out_conv(x)
        x = nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


def align_logits_and_target_spatial(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop logits/target to a common spatial size.

    This keeps training robust when model output spatial size differs slightly
    from labels due to pooling/upsampling rounding.
    """
    if logits.shape[-2:] == target.shape[-2:]:
        return logits, target

    h = min(logits.shape[-2], target.shape[-2])
    w = min(logits.shape[-1], target.shape[-1])

    def _crop4d(t: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
        th, tw = t.shape[-2], t.shape[-1]
        sh = max((th - ch) // 2, 0)
        sw = max((tw - cw) // 2, 0)
        return t[:, :, sh : sh + ch, sw : sw + cw]

    def _crop3d(t: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
        th, tw = t.shape[-2], t.shape[-1]
        sh = max((th - ch) // 2, 0)
        sw = max((tw - cw) // 2, 0)
        return t[:, sh : sh + ch, sw : sw + cw]

    return _crop4d(logits, h, w), _crop3d(target, h, w)


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 19,
        eps: float = 1e-6,
        background_index: int = 0,
        ignore_background: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.background_index = background_index
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = align_logits_and_target_spatial(logits, target)
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
        valid_mask = (target != self.ignore_index).float()  # (B,H,W)

        losses = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            if self.ignore_background and cls == self.background_index:
                continue
            tgt = (target == cls).float()
            # Skip classes absent in this batch to avoid noisy gradients on long-tail labels.
            if tgt.sum().item() <= 0:
                continue
            p = probs[:, cls, :, :] * valid_mask
            t = tgt * valid_mask
            inter = (p * t).sum()
            denom = (p * p).sum() + (t * t).sum()
            dice = (2.0 * inter + self.eps) / (denom + self.eps)
            losses.append(1.0 - dice)

        if not losses:
            # Keep zero-loss connected to autograd graph to avoid backward() errors.
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class FocalLossWithSampling(nn.Module):
    """多类 Focal Loss + patch 内 0/1/2 类像素有放回重采样。

    对于每个样本：
      1. 仅在 focus_classes (默认 [0,1,2]) 且非 ignore_index 的像素上工作；
      2. 统计各类像素数，取其中最大值 n_max；
      3. 对每一类有放回采样 n_max 个像素；
      4. 在这些采样像素上计算 Focal Loss。
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        ignore_index: int = 255,
        focus_classes: Optional[List[int]] = None,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        if focus_classes is None:
            focus_classes = [0, 1, 2]
        self.focus_classes = focus_classes

        # 可选：为不同类别指定 Focal Loss 权重（例如提高 class 1 的权重）。
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() < num_classes:
                pad = torch.ones(num_classes - w.numel(), dtype=torch.float32)
                w = torch.cat([w, pad], dim=0)
            elif w.numel() > num_classes:
                w = w[:num_classes]
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = align_logits_and_target_spatial(logits, target)
        B, C, H, W = logits.shape
        device = logits.device

        logits_flat = logits.view(B, C, -1).permute(0, 2, 1)  # (B,N,C)
        target_flat = target.reshape(B, -1)  # (B,N)

        focus = [
            c
            for c in self.focus_classes
            if 0 <= c < self.num_classes and c != self.ignore_index
        ]

        # 若 focus_classes 无效，则退化为所有合法像素上的标准 Focal Loss。
        if not focus:
            all_logits = logits_flat.reshape(-1, C)
            all_target = target_flat.reshape(-1)
            # 仅在合法类别上计算：排除 ignore_index 以及越界标签
            valid_mask = (
                (all_target != self.ignore_index)
                & (all_target >= 0)
                & (all_target < self.num_classes)
            )
            if not valid_mask.any():
                return logits.sum() * 0.0
            all_logits = all_logits[valid_mask]
            all_target = all_target[valid_mask]
            log_probs = nn.functional.log_softmax(all_logits, dim=1)
            ce = nn.functional.nll_loss(log_probs, all_target, reduction="none")
            pt = torch.exp(-ce)

            if self.class_weights is not None:
                w = self.class_weights[all_target]
                loss = ((1.0 - pt) ** self.gamma) * ce * w
            else:
                loss = ((1.0 - pt) ** self.gamma) * ce
            return loss.mean()

        selected_logits: List[torch.Tensor] = []
        selected_targets: List[torch.Tensor] = []

        for b in range(B):
            y_b = target_flat[b]  # (N,)
            valid_mask = y_b != self.ignore_index
            if not valid_mask.any():
                continue

            idx_valid = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            y_valid = y_b[valid_mask]

            class_indices = {}
            max_count = 0
            for cls in focus:
                idx_cls = idx_valid[y_valid == cls]
                if idx_cls.numel() > 0:
                    class_indices[cls] = idx_cls
                    if idx_cls.numel() > max_count:
                        max_count = idx_cls.numel()

            if max_count == 0:
                continue

            sampled_pos_list = []
            for cls, idx_cls in class_indices.items():
                # 对该类的像素做有放回采样，使每类像素数约为 max_count。
                if idx_cls.numel() == 1:
                    sampled = idx_cls.expand(max_count)
                else:
                    rand_idx = torch.randint(0, idx_cls.numel(), (max_count,), device=device)
                    sampled = idx_cls[rand_idx]
                sampled_pos_list.append(sampled)

            if not sampled_pos_list:
                continue

            sampled_pos = torch.cat(sampled_pos_list, dim=0)

            # 保险起见，再做一次索引范围过滤，防止任何越界索引进入 CUDA 内核。
            N = logits_flat.shape[1]
            in_bounds = (sampled_pos >= 0) & (sampled_pos < N)
            if not in_bounds.any():
                continue
            sampled_pos = sampled_pos[in_bounds]

            selected_logits.append(logits_flat[b, sampled_pos, :])
            selected_targets.append(y_b[sampled_pos])

        if not selected_logits:
            return logits.sum() * 0.0

        logits_sel = torch.cat(selected_logits, dim=0)
        target_sel = torch.cat(selected_targets, dim=0)

        # 再次过滤，确保标签在 [0, num_classes-1] 且不为 ignore_index，避免越界索引
        valid = (
            (target_sel != self.ignore_index)
            & (target_sel >= 0)
            & (target_sel < self.num_classes)
        )
        if not valid.any():
            return logits.sum() * 0.0
        logits_sel = logits_sel[valid]
        target_sel = target_sel[valid]

        log_probs = nn.functional.log_softmax(logits_sel, dim=1)
        ce = nn.functional.nll_loss(log_probs, target_sel, reduction="none")
        pt = torch.exp(-ce)

        if self.class_weights is not None:
            class_weights = self.class_weights
            if class_weights.device != target_sel.device:
                class_weights = class_weights.to(target_sel.device)
            w = class_weights[target_sel]
            loss = ((1.0 - pt) ** self.gamma) * ce * w
        else:
            loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def foreground_aux_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    background_index: int,
    ignore_index: int,
) -> torch.Tensor:
    """Binary foreground loss from multi-class probabilities.

    Encourages the model to separate foreground vs background first, which helps
    avoid early collapse to all-background predictions.
    """
    probs = torch.softmax(logits, dim=1)
    n_classes = probs.shape[1]
    fg_classes = [c for c in range(n_classes) if c not in (background_index, ignore_index)]
    if not fg_classes:
        return logits.sum() * 0.0

    fg_prob = probs[:, fg_classes, :, :].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
    valid = (target != ignore_index)
    fg_tgt = ((target != background_index) & valid).float()

    bce = -(fg_tgt * torch.log(fg_prob) + (1.0 - fg_tgt) * torch.log(1.0 - fg_prob))
    denom = valid.float().sum().clamp_min(1.0)
    return (bce * valid.float()).sum() / denom


def background_ratio_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    background_index: int,
    ignore_index: int,
) -> torch.Tensor:
    """Match predicted background probability ratio to label background ratio."""
    probs_bg = torch.softmax(logits, dim=1)[:, background_index, :, :]
    valid = (target != ignore_index).float()
    denom = valid.sum().clamp_min(1.0)

    pred_bg_ratio = (probs_bg * valid).sum() / denom
    true_bg_ratio = (((target == background_index).float()) * valid).sum() / denom
    return (pred_bg_ratio - true_bg_ratio).pow(2)


def estimate_class_weights(
    dataset,
    num_classes: int,
    ignore_index: int,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Estimate inverse-frequency class weights from training subset labels."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for _x, y in loader:
        y_flat = y.reshape(-1)
        # 仅统计合法类别 [0, num_classes-1]，并排除 ignore_index；
        # 对于 255 等越界值，自动视为 ignore，避免 bincount 长度>num_classes。
        valid = (y_flat != ignore_index) & (y_flat >= 0) & (y_flat < num_classes)
        if valid.any():
            binc = torch.bincount(y_flat[valid], minlength=num_classes).to(torch.float64)
            counts += binc

    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32)
    if present.any():
        total = counts[present].sum()
        weights[present] = (total / counts[present]).to(torch.float32)
        # Normalize so average weight over present classes is ~1
        weights[present] = weights[present] / weights[present].mean().clamp_min(1e-6)
        weights[present] = weights[present].clamp(min=min_weight, max=max_weight)

    if 0 <= ignore_index < num_classes:
        weights[ignore_index] = 0.0

    return weights


def compute_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    background_index: int = 0,
    ignore_background: bool = True,
) -> tuple[float, float]:
    """Return (pixel_acc, mean_iou) on valid pixels."""
    logits, target = align_logits_and_target_spatial(logits, target)
    preds = logits.argmax(dim=1)
    mask = target != ignore_index
    valid = int(mask.sum().item())
    if valid == 0:
        return 0.0, 0.0

    pixel_acc = float((preds[mask] == target[mask]).sum().item()) / float(valid)

    ious: List[float] = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if ignore_background and cls == background_index:
            continue
        p = (preds == cls) & mask
        t = (target == cls) & mask
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(float(inter) / float(union))

    miou = float(np.mean(ious)) if ious else 0.0
    return pixel_acc, miou


def compute_confusion_matrix(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """Compute a raw confusion matrix over valid pixels."""
    logits, target = align_logits_and_target_spatial(logits, target)
    preds = logits.argmax(dim=1)

    mask = target != ignore_index
    if int(mask.sum().item()) == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    y_true = target[mask].reshape(-1).to(torch.int64)
    y_pred = preds[mask].reshape(-1).to(torch.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    if int(valid.sum().item()) == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    idx = y_true[valid] * num_classes + y_pred[valid]
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes).cpu().numpy().astype(np.int64)


def compute_f1_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    background_index: int = 0,
    ignore_background: bool = True,
    class_indices: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    """Return per-class precision/recall/F1, macro averages, weighted F1, and class weights.

    The confusion matrix is expected to be shaped (num_classes, num_classes)
    with rows = ground truth and columns = predictions.
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape {cm.shape}")

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    if class_indices is None:
        valid_classes = [cls for cls in range(cm.shape[0]) if not (ignore_background and cls == background_index)]
    else:
        valid_classes = []
        for cls in class_indices:
            if not (0 <= cls < cm.shape[0]):
                continue
            if ignore_background and cls == background_index:
                continue
            valid_classes.append(cls)
    if valid_classes:
        macro_precision = float(np.mean(precision[valid_classes]))
        macro_recall = float(np.mean(recall[valid_classes]))
        macro_f1 = float(np.mean(f1[valid_classes]))
        class_support = cm.sum(axis=1)
        support = class_support[valid_classes]
        support_sum = float(np.sum(support))
        if support_sum > 0.0:
            weighted_f1 = float(np.sum(f1[valid_classes] * support) / support_sum)
            class_weights = np.zeros(cm.shape[0], dtype=np.float64)
            class_weights[valid_classes] = support / support_sum
        else:
            weighted_f1 = 0.0
            class_weights = np.zeros(cm.shape[0], dtype=np.float64)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0
        class_weights = np.zeros(cm.shape[0], dtype=np.float64)

    return precision, recall, f1, macro_precision, macro_recall, macro_f1, weighted_f1, class_weights


def save_confusion_matrix_visualization(
    confusion_matrix: np.ndarray,
    output_dir: Path,
    prefix: str = "val",
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    """Save confusion matrix as NPY and PNG visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got shape {cm.shape}")

    np.save(output_dir / f"{prefix}_confusion_matrix.npy", cm.astype(np.int64))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    np.save(output_dir / f"{prefix}_confusion_matrix_norm.npy", cm_norm.astype(np.float32))

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


def evaluate_per_class_iou(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    background_index: int = 0,
    ignore_background: bool = True,
) -> np.ndarray:
    """Compute per-class IoU on a dataset.

    Returns a numpy array of length ``num_classes`` where entries for
    ignored classes (``ignore_index`` and optionally ``background_index``)
    are set to NaN. Classes that are not ignored but have zero pixels will
    return 0.0 (rather than NaN) so they are included in downstream
    averages such as mF1/weighted-F1.
    """

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logits, y = align_logits_and_target_spatial(logits, y)
            preds = logits.argmax(dim=1)

            mask = y != ignore_index

            for cls in range(num_classes):
                if cls == ignore_index:
                    continue
                if ignore_background and cls == background_index:
                    continue

                p = (preds == cls) & mask
                t = (y == cls) & mask
                inter[cls] += (p & t).sum().item()
                union[cls] += (p | t).sum().item()

    per_class_iou = np.full(num_classes, 0.0, dtype=np.float32)
    for cls in range(num_classes):
        if cls == ignore_index:
            per_class_iou[cls] = np.nan
            continue
        if ignore_background and cls == background_index:
            per_class_iou[cls] = np.nan
            continue

        if union[cls] > 0.0:
            per_class_iou[cls] = float(inter[cls] / union[cls])
        else:
            # No pixels for this class in GT+pred: report IoU=0.0 (include in averages)
            per_class_iou[cls] = 0.0

    return per_class_iou


def dump_per_class_iou_report(
    per_class_iou: np.ndarray,
    output_dir: Path,
    prefix: str = "best_val",
    per_class_f1: Optional[np.ndarray] = None,
    weighted_f1: Optional[float] = None,
) -> List[dict]:
    """Print and save per-class IoU report as NPY/JSON/CSV.

    Returns a list of dict rows: {class_index, iou, f1?, weighted_f1?}.
    """
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


def labels_to_rgb(mask, num_classes, ignore_index):
    cmap = plt.get_cmap("tab20", num_classes)
    palette = cmap(np.arange(num_classes))[:, :3]
    safe_mask = np.clip(mask, 0, num_classes - 1)
    rgb = palette[safe_mask]
    # 无论 ignore_index 是否 < num_classes，都单独覆盖颜色
    rgb[mask == ignore_index] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return rgb


def save_val_visualizations(
    model: nn.Module,
    val_ds,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
    ignore_index: int,
    max_images: int = 4,
) -> None:
    if val_ds is None or len(val_ds) == 0 or max_images <= 0:
        return

    vis_dir = output_dir / "val_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n = min(max_images, len(val_ds))
    with torch.no_grad():
        for i in range(n):
            x, y = val_ds[i]
            logits = model(x.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            x_np = x.cpu().numpy()
            rgb = x_np[:3] if x_np.shape[0] >= 3 else np.repeat(x_np[:1], 3, axis=0)
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

            y_np = y.cpu().numpy().astype(np.int64)
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


def load_unet_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device, strict: bool = True) -> None:
    """Load a saved UNet checkpoint for fine-tuning."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if strict:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint (strict) from {checkpoint_path}")
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
        return

    model_state = model.state_dict()
    filtered_state = {}
    skipped_mismatch = []
    for k, v in state_dict.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Loaded checkpoint (non-strict, shape-filtered) from {checkpoint_path} "
        f"with {len(filtered_state)}/{len(model_state)} compatible tensor(s)."
    )
    if skipped_mismatch:
        print(
            f"Skipped shape-mismatched keys ({len(skipped_mismatch)}), "
            f"example: {skipped_mismatch[:3]}"
        )
    if missing:
        print(f"Missing keys after filtered load ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")


# ---------- Training script ----------


def train(args: argparse.Namespace):
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
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Train dataset (always来自 embeddings_path / labels_file)
    per_patch_labels_train = bool(args.per_patch_labels or labels_path.is_dir())
    if labels_path.is_dir() and not args.per_patch_labels:
        print(
            "Detected train labels directory; enabling per-patch label matching automatically "
            "(equivalent to --per_patch_labels)."
        )

    full_dataset = EmbeddingSegmentationDataset(
        emb_path,
        labels_path,
        per_patch_labels=per_patch_labels_train,
        embedding_key=args.embedding_key,
        resample_size=args.resample_size,
    )

    # Peek at one train sample to infer channel and spatial sizes.
    sample_feats, sample_labels = full_dataset[0]
    C, H, W = sample_feats.shape
    print(f"Full dataset size: {len(full_dataset)} samples")
    print(f"Sample features shape (C,H,W)=({C},{H},{W}), labels shape (H,W)=({H},{W})")

    # Fail fast if labels are not class indices (common mistake: using raw ParcelIDs_*.npy).
    valid_ratio, y_min, y_max = estimate_label_valid_ratio(
        full_dataset,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_samples=16,
    )
    if valid_ratio < 0.95:
        sample_unique = torch.unique(sample_labels.reshape(-1)).cpu().numpy().tolist()
        sample_unique_preview = sample_unique[:12]
        raise ValueError(
            "Label sanity check failed: most label pixels are outside valid class range. "
            f"valid_ratio={valid_ratio:.4f}, min={y_min}, max={y_max}, "
            f"num_classes={args.num_classes}, ignore_index={args.ignore_index}, "
            f"sample_unique_head={sample_unique_preview}. "
            "This usually means labels are raw parcel IDs (e.g., ParcelIDs_XXXXX.npy) rather than class indices. "
            "Please convert labels to class-index masks first (labels in [0, num_classes-1], plus ignore_index), "
            "or set the correct --ignore_index if your void label is not the current value."
        )

    # Train/val split 或使用单独的验证集
    raw_base_train_ds = None
    base_train_ds = None
    base_val_ds = None

    if args.val_embeddings_path and args.val_labels_file:
        # 使用独立的验证集：val_embeddings_path / val_labels_file
        val_emb_path = Path(args.val_embeddings_path)
        val_labels_path = Path(args.val_labels_file)
        if not val_emb_path.exists():
            raise FileNotFoundError(f"Val embeddings path not found: {val_emb_path}")
        if not val_labels_path.exists():
            raise FileNotFoundError(f"Val labels file not found: {val_labels_path}")

        per_patch_labels_val = bool(args.per_patch_labels or val_labels_path.is_dir())
        if val_labels_path.is_dir() and not args.per_patch_labels:
            print(
                "Detected val labels directory; enabling per-patch label matching automatically "
                "(equivalent to --per_patch_labels)."
            )

        val_full_dataset = EmbeddingSegmentationDataset(
            val_emb_path,
            val_labels_path,
            per_patch_labels=per_patch_labels_val,
            embedding_key=args.embedding_key,
            resample_size=args.resample_size,
        )

        # 训练集全部样本都用于训练；warmup 等逻辑依旧使用 train_idx=range(len(full_dataset))
        n_total = len(full_dataset)
        train_idx = np.arange(n_total, dtype=int)
        raw_base_train_ds = Subset(full_dataset, train_idx.tolist())
        base_train_ds = raw_base_train_ds
        base_val_ds = val_full_dataset
        print(
            f"Using external validation set: train_samples={len(base_train_ds)}, "
            f"val_samples={len(base_val_ds)}"
        )
    else:
        # 默认行为：在单一数据集上按 val_fraction 做随机划分
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * args.val_fraction)) if n_total > 1 else 0
        n_train = n_total - n_val
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:] if n_val > 0 else np.array([], dtype=int)

        raw_base_train_ds = Subset(full_dataset, train_idx.tolist())
        base_train_ds = raw_base_train_ds
        base_val_ds = Subset(full_dataset, val_idx.tolist()) if n_val > 0 else None

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
        if base_val_ds is not None:
            base_val_ds = FeatureNormalizeDataset(base_val_ds, feat_mean, feat_std)
    else:
        print("Feature normalization: OFF")

    train_ds = base_train_ds
    val_ds = base_val_ds

    if args.enable_augmentation == 1 and len(base_train_ds) > 0:
        train_fg_ratios = estimate_sample_foreground_ratios(
            raw_base_train_ds,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )
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
            sample_foreground_ratios=train_fg_ratios,
            fg_ratio_threshold=args.fg_aug_ratio_threshold,
            fg_aug_boost=args.fg_aug_boost,
        )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds is not None else 0}")
    if args.enable_augmentation == 1:
        print(
            "Train augmentation: ON "
            f"(repeat={args.train_repeat_factor}, flip_prob={args.aug_flip_prob}, "
            f"rot90_prob={args.aug_rot90_prob}, noise_std={args.aug_noise_std}, "
            f"gain_std={args.aug_gain_std}, crop_size={args.train_crop_size}, "
            f"fg_crop_prob={args.fg_crop_prob}, fg_ratio_threshold={args.fg_aug_ratio_threshold}, "
            f"fg_aug_boost={args.fg_aug_boost})"
        )
    else:
        print("Train augmentation: OFF")

    train_sampler = None
    if args.use_fg_sampler == 1 and len(raw_base_train_ds) > 0:
        fg_ratios = estimate_sample_foreground_ratios(
            raw_base_train_ds,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )
        weights = args.fg_sampler_min_weight + np.power(
            np.clip(fg_ratios, 1e-6, None),
            args.fg_sampler_power,
        )
        if len(train_ds) != len(raw_base_train_ds):
            rep = int(np.ceil(len(train_ds) / len(raw_base_train_ds)))
            weights = np.tile(weights, rep)[:len(train_ds)]
        weights_t = torch.as_tensor(weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=weights_t,
            num_samples=len(train_ds),
            replacement=True,
        )
        print(
            f"Foreground sampler: ON (power={args.fg_sampler_power}, min_weight={args.fg_sampler_min_weight})"
        )
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

    warmup_loader = None
    if args.warmup_epochs > 0 and len(raw_base_train_ds) > 1:
        fg_ratios = estimate_sample_foreground_ratios(
            raw_base_train_ds,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )
        k = max(1, int(len(raw_base_train_ds) * args.warmup_top_fg_fraction))
        top_local = np.argsort(-fg_ratios)[:k]
        # Map local subset indices back to full_dataset indices.
        warmup_global_idx = train_idx[top_local]
        warmup_raw_ds = Subset(full_dataset, warmup_global_idx.tolist())
        warmup_ds = warmup_raw_ds
        if args.normalize_features == 1 and len(base_train_ds) > 0:
            warmup_ds = FeatureNormalizeDataset(warmup_raw_ds, feat_mean, feat_std)
        warmup_loader = DataLoader(
            warmup_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(
            f"Warmup curriculum: ON (epochs={args.warmup_epochs}, top_fg_fraction={args.warmup_top_fg_fraction}, "
            f"warmup_samples={len(warmup_ds)})"
        )
    else:
        print("Warmup curriculum: OFF")
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

    if args.model_variant == "resse":
        model = UNetResSE(
            in_channels=C,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
            depth=args.unet_depth,
            dropout=args.dropout_rate,
        )
        print(
            f"Model: UNetResSE(depth={args.unet_depth}, base_channels={args.base_channels}, "
            f"dropout={args.dropout_rate})"
        )
    elif args.model_variant == "lightweight":
        model = UNetLightweight(
            in_channels=C,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
            depth=args.unet_depth,
        )
        print(
            f"Model: UNetLightweight(depth={args.unet_depth}, base_channels={args.base_channels})"
        )
    elif args.model_variant == "aspp":
        model = DeepLabLite(
            in_channels=C,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
        )
        print(f"Model: DeepLabLite(base_channels={args.base_channels})")
    elif args.model_variant == "from_embeddings":
        # dynamic import to avoid circular import at module load time
        from .unet_decoder import UNetFromEmbeddings

        model = UNetFromEmbeddings(
            embedding_channels=C,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            norm=args.norm_type,
            freeze_encoder=bool(args.freeze_encoder),
        )
        print(f"Model: UNetFromEmbeddings(base_channels={args.base_channels}, freeze_encoder={args.freeze_encoder})")
    else:
        if args.unet_depth == 5:
            model = UNetDeep(
                in_channels=C,
                num_classes=args.num_classes,
                base_ch=args.base_channels,
                norm=args.norm_type,
            )
            print(f"Model: UNetDeep(depth=5, base_channels={args.base_channels})")
        else:
            model = UNet(
                in_channels=C,
                num_classes=args.num_classes,
                base_ch=args.base_channels,
                norm=args.norm_type,
            )
            print(f"Model: UNet(depth=4, base_channels={args.base_channels})")
    model.to(device)

    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        load_unet_checkpoint(
            model,
            checkpoint_path=ckpt_path,
            device=device,
            strict=bool(args.resume_strict),
        )

    # 损失：Focal Loss（带 0/1/2 类像素重采样） + Dice Loss。
    focal_class_weights = None
    if getattr(args, "focal_class_weights", ""):
        try:
            focal_class_weights = [
                float(x)
                for x in str(args.focal_class_weights).split(",")
                if x.strip() != ""
            ]
            print(f"Using focal class weights: {focal_class_weights}")
        except ValueError:
            print(
                f"Warning: could not parse --focal_class_weights='{args.focal_class_weights}', "
                "falling back to uniform weights."
            )
            focal_class_weights = None

    # Focal 采样类别：默认使用全部前景类（排除 background 与 ignore）。
    # 注意：'all' 应包含 background，'all_fg' 才排除 background。
    focus_classes: List[int]
    focus_spec = str(getattr(args, "focal_focus_classes", "all_fg")).strip().lower()
    if focus_spec in ("", "all_fg"):
        focus_classes = [
            c
            for c in range(args.num_classes)
            if c not in (args.background_index, args.ignore_index)
        ]
    elif focus_spec == "all":
        focus_classes = [c for c in range(args.num_classes) if c != args.ignore_index]
    else:
        parsed: List[int] = []
        for tok in str(args.focal_focus_classes).split(","):
            tok = tok.strip()
            if tok == "":
                continue
            try:
                parsed.append(int(tok))
            except ValueError:
                continue
        focus_classes = [
            c
            for c in parsed
            if 0 <= c < args.num_classes and c != args.ignore_index
        ]
    if not focus_classes:
        # 兜底：至少使用所有合法类别，避免退化为空。
        focus_classes = [c for c in range(args.num_classes) if c != args.ignore_index]
    print(f"Focal focus classes: {focus_classes}")

    criterion_focal = FocalLossWithSampling(
        num_classes=args.num_classes,
        gamma=args.focal_gamma,
        ignore_index=args.ignore_index,
        focus_classes=focus_classes,
        class_weights=focal_class_weights,
    )

    # CE provides dense per-pixel supervision and stabilizes optimization.
    ce_class_weights = estimate_class_weights(
        raw_base_train_ds,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        min_weight=args.class_weight_min,
        max_weight=args.class_weight_max,
    )
    if 0 <= args.background_index < args.num_classes:
        ce_class_weights[args.background_index] = min(
            float(ce_class_weights[args.background_index].item()),
            float(args.background_ce_weight),
        )
    print(f"CE class weights: {ce_class_weights.tolist()}")
    criterion_ce = nn.CrossEntropyLoss(
        weight=ce_class_weights.to(device),
        ignore_index=args.ignore_index,
        label_smoothing=max(0.0, float(args.label_smoothing)),
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
    best_val_weighted_f1 = 0.0
    best_val_mf1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        running_miou_sum = 0.0
        running_miou_count = 0
        running_pred_bg = 0
        running_pred_total = 0

        active_loader = warmup_loader if (warmup_loader is not None and epoch < args.warmup_epochs) else train_loader

        fg_aux_scale = 0.0
        if args.fg_aux_weight > 0:
            if args.fg_aux_warmup_epochs > 0:
                fg_aux_scale = min(1.0, float(epoch + 1) / float(args.fg_aux_warmup_epochs))
            else:
                fg_aux_scale = 1.0
        fg_aux_scale *= float(args.fg_aux_weight)

        bg_ratio_scale = 0.0
        if args.bg_ratio_weight > 0:
            if args.bg_ratio_warmup_epochs > 0:
                bg_ratio_scale = min(1.0, float(epoch + 1) / float(args.bg_ratio_warmup_epochs))
            else:
                bg_ratio_scale = 1.0
        bg_ratio_scale *= float(args.bg_ratio_weight)

        for x, y in active_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B,num_classes,H,W)
            logits, y = align_logits_and_target_spatial(logits, y)
            loss_ce = criterion_ce(logits, y)
            loss_focal = criterion_focal(logits, y)
            loss_dice = criterion_dice(logits, y)
            loss_fg_aux = foreground_aux_loss(
                logits,
                y,
                background_index=args.background_index,
                ignore_index=args.ignore_index,
            )
            loss_bg_ratio = background_ratio_loss(
                logits,
                y,
                background_index=args.background_index,
                ignore_index=args.ignore_index,
            )
            loss = (
                loss_ce
                + loss_focal
                + args.dice_weight * loss_dice
                + fg_aux_scale * loss_fg_aux
                + bg_ratio_scale * loss_bg_ratio
            )
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
                running_total += total
                pred = logits.argmax(dim=1)
                running_pred_bg += int(((pred == args.background_index) & mask).sum().item())
                running_pred_total += int(total)
                running_miou_sum += miou
                running_miou_count += 1

        train_loss = running_loss / max(1, len(active_loader.dataset))
        train_acc = running_correct / max(1, running_total) if running_total > 0 else 0.0
        train_miou = running_miou_sum / max(1, running_miou_count)
        train_pred_bg_ratio = running_pred_bg / max(1, running_pred_total)

        # Validation
        val_loss = None
        val_acc = None
        val_miou = None
        val_weighted_f1 = None
        val_mf1 = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_correct = 0
            val_running_total = 0
            val_running_miou_sum = 0.0
            val_running_miou_count = 0
            val_running_pred_bg = 0
            val_running_pred_total = 0
            val_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    logits, y = align_logits_and_target_spatial(logits, y)
                    loss_ce = criterion_ce(logits, y)
                    loss_focal = criterion_focal(logits, y)
                    loss_dice = criterion_dice(logits, y)
                    loss_fg_aux = foreground_aux_loss(
                        logits,
                        y,
                        background_index=args.background_index,
                        ignore_index=args.ignore_index,
                    )
                    loss_bg_ratio = background_ratio_loss(
                        logits,
                        y,
                        background_index=args.background_index,
                        ignore_index=args.ignore_index,
                    )
                    loss = (
                        loss_ce
                        + loss_focal
                        + args.dice_weight * loss_dice
                        + fg_aux_scale * loss_fg_aux
                        + bg_ratio_scale * loss_bg_ratio
                    )
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
                    val_running_total += total
                    pred = logits.argmax(dim=1)
                    val_running_pred_bg += int(((pred == args.background_index) & mask).sum().item())
                    val_running_pred_total += int(total)
                    val_running_miou_sum += miou
                    val_running_miou_count += 1
                    val_confusion += compute_confusion_matrix(
                        logits,
                        y,
                        num_classes=args.num_classes,
                        ignore_index=args.ignore_index,
                    )

            val_loss = val_running_loss / max(1, len(val_ds))
            val_acc = (
                val_running_correct / max(1, val_running_total)
                if val_running_total > 0
                else 0.0
            )
            val_miou = val_running_miou_sum / max(1, val_running_miou_count)
            val_pred_bg_ratio = val_running_pred_bg / max(1, val_running_pred_total)
            (
                _precision,
                _recall,
                _f1,
                _macro_precision,
                _macro_recall,
                macro_f1,
                weighted_f1,
                _class_weights,
            ) = compute_f1_from_confusion_matrix(
                val_confusion,
                background_index=args.background_index,
                ignore_background=args.ignore_background_in_metrics,
            )
            val_weighted_f1 = weighted_f1
            val_mf1 = macro_f1

            # Track best model by mIoU first, then val_loss as tie-breaker.
            improved = (val_miou > best_val_miou) or (
                np.isclose(val_miou, best_val_miou) and val_loss < best_val_loss
            )
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_miou = val_miou
                best_val_weighted_f1 = val_weighted_f1
                best_val_mf1 = val_mf1
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if val_loss is not None:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
                f"train_pred_bg: {train_pred_bg_ratio:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mIoU: {val_miou:.4f}, "
                f"val_mF1: {val_mf1:.4f}, val_weighted_F1: {val_weighted_f1:.4f}, "
                f"val_pred_bg: {val_pred_bg_ratio:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
                f"train_pred_bg: {train_pred_bg_ratio:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_loss is not None:
            if args.scheduler_monitor == "val_miou":
                scheduler.step(val_miou)
            else:
                scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(no mIoU improvement for {args.early_stop_patience} epoch(s))."
            )
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save latest model (filename reflects model variant)
    ckpt_latest = out_dir / f"{args.model_variant}_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "in_channels": C}, ckpt_latest)
    print(f"Saved latest model checkpoint to {ckpt_latest}")

    # Save best model (by validation)
    if best_state is not None:
        ckpt_best = out_dir / f"{args.model_variant}_best.pt"
        torch.save({"model_state_dict": best_state, "in_channels": C}, ckpt_best)
        print(
            f"Saved best model checkpoint to {ckpt_best} "
            f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, val_mIoU={best_val_miou:.4f})"
        )

    # Export a few validation examples for qualitative inspection and
    # compute per-class IoU on the best checkpoint (if available).
    if best_state is not None:
        model.load_state_dict(best_state)

        if val_loader is not None and val_ds is not None and len(val_ds) > 0:
            per_class_iou = evaluate_per_class_iou(
                model=model,
                data_loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                background_index=args.background_index,
                ignore_background=args.ignore_background_in_metrics,
            )

            dump_per_class_iou_report(per_class_iou, output_dir=out_dir, prefix="best_val")

    save_val_visualizations(
        model=model,
        val_ds=val_ds,
        device=device,
        output_dir=out_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.save_val_samples,
    )
    if val_ds is not None and len(val_ds) > 0 and args.save_val_samples > 0:
        print(
            f"Saved {min(args.save_val_samples, len(val_ds))} validation visualization(s) to "
            f"{out_dir / 'val_visualizations'}"
        )

    # 返回本次训练的关键验证指标，便于在外部脚本中做超参搜索对比。
    return {
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "best_val_miou": float(best_val_miou),
        "best_val_weighted_f1": float(best_val_weighted_f1),
        "best_val_mf1": float(best_val_mf1),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a simple U-Net segmentation model using AEF embeddings "
            "as input features and integer label mask as targets."
        )
    )
    p.add_argument(
        "--embeddings_path",
        type=str,
        default="",
        help=(
            "Path to embeddings. If a file, uses a single sample. If a "
            "directory, uses all *.npy files inside as separate samples."
        ),
    )
    p.add_argument(
        "--embedding_key",
        type=str,
        default="auto",
        help="Embedding key in .npy object array to use (auto/embeddings/embeddings_native/embeddings_per_time)",
    )
    p.add_argument(
        "--resample_size",
        type=int,
        default=0,
        help="If >0, resize each embedding/label sample to this square size before training (e.g., 256).",
    )
    p.add_argument(
        "--labels_file",
        type=str,
        default="",
        help=(
            "Labels file (.npy integer mask). "
            "If this is a directory, per-patch matching is enabled automatically; "
            "you can also force it with --per_patch_labels. In that case it should "
            "be a directory containing per-patch files named ParcelIDs_XXXXX.npy."
        ),
    )
    p.add_argument(
        "--val_embeddings_path",
        type=str,
        default="",
        help=(
            "Optional: separate validation embeddings path. If set together with "
            "--val_labels_file, this directory/file will be used as validation set "
            "instead of splitting --embeddings_path by --val_fraction."
        ),
    )
    p.add_argument(
        "--val_labels_file",
        type=str,
        default="",
        help=(
            "Optional: validation labels file or directory corresponding to "
            "--val_embeddings_path."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save checkpoints and logs",
    )
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training and validation (default 4)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (default 4)",
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
        default=255,
        help="Label index to ignore in loss (e.g., 'Void label' = 255)",
    )
    p.add_argument(
        "--background_index",
        type=int,
        default=0,
        help="Background class index (default 0).",
    )
    p.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base number of channels in U-Net encoder (default 32)",
    )
    p.add_argument(
        "--unet_depth",
        type=int,
        default=4,
        choices=[4, 5],
        help="UNet depth: 4 (default) or 5 (deeper)",
    )
    p.add_argument(
        "--model_variant",
        type=str,
        default="basic",
        choices=["basic", "resse", "lightweight", "aspp", "from_embeddings"],
        help="Segmentation model variant: basic VGG16-style UNet, resse UNet, lightweight UNet, or aspp DeepLabLite",
    )
    p.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout rate used by resse variant blocks (default 0.1)",
    )
    p.add_argument(
        "--norm_type",
        type=str,
        default="instance",
        choices=["instance", "group", "batch"],
        help="Normalization layer type in UNet blocks: instance, group, or batch (default instance)",
    )
    p.add_argument(
        "--freeze_encoder",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, freeze encoder weights when using from_embeddings variant (default 1)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file to override args",
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
            "files named ParcelIDs_XXXXX.npy that correspond 1:1 to "
            "embedding_XXXXX*.npy files in --embeddings_path."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Path to a saved UNet checkpoint (.pt) to resume/fine-tune from",
    )
    p.add_argument(
        "--resume_strict",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, strictly match checkpoint keys when loading (default 1)",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer (default 1e-4)",
    )
    p.add_argument(
        "--enable_augmentation",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, apply train-time data augmentation (default 1).",
    )
    p.add_argument(
        "--normalize_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, apply channel-wise feature normalization using train-set stats (default 1)",
    )
    p.add_argument(
        "--feature_stats_max_samples",
        type=int,
        default=0,
        help="Max train samples to estimate feature mean/std (0 means all, default 0)",
    )
    p.add_argument(
        "--train_repeat_factor",
        type=int,
        default=3,
        help="Virtual expansion factor for train set via repeated sampling (default 3)",
    )
    p.add_argument(
        "--aug_flip_prob",
        type=float,
        default=0.5,
        help="Probability for each random horizontal/vertical flip (default 0.5)",
    )
    p.add_argument(
        "--aug_rot90_prob",
        type=float,
        default=0.5,
        help="Probability of random 90-degree rotation (default 0.5)",
    )
    p.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.01,
        help="Gaussian noise std added to embeddings (default 0.01)",
    )
    p.add_argument(
        "--aug_gain_std",
        type=float,
        default=0.05,
        help="Global multiplicative gain jitter std for embeddings (default 0.05)",
    )
    p.add_argument(
        "--train_crop_size",
        type=int,
        default=96,
        help="Random crop size for training augmentation (0 disables, default 96)",
    )
    p.add_argument(
        "--fg_crop_prob",
        type=float,
        default=0.8,
        help="Probability to center random crop on a foreground pixel (default 0.8)",
    )
    p.add_argument(
        "--fg_aug_ratio_threshold",
        type=float,
        default=0.2,
        help=(
            "Foreground-ratio threshold above which a sample receives stronger augmentation "
            "(default 0.2)."
        ),
    )
    p.add_argument(
        "--fg_aug_boost",
        type=float,
        default=1.5,
        help=(
            "Maximum augmentation boost applied to samples whose foreground ratio is above "
            "--fg_aug_ratio_threshold (default 1.5)."
        ),
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=6,
        help="Number of initial epochs trained on foreground-rich subset (0 disables, default 6)",
    )
    p.add_argument(
        "--warmup_top_fg_fraction",
        type=float,
        default=0.35,
        help="Top fraction of train samples by foreground ratio for warmup subset (default 0.35)",
    )
    p.add_argument(
        "--use_fg_sampler",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, use foreground-ratio weighted sampling for training batches (default 1)",
    )
    p.add_argument(
        "--fg_sampler_power",
        type=float,
        default=1.5,
        help="Exponent for foreground-ratio sample weights (default 1.5)",
    )
    p.add_argument(
        "--fg_sampler_min_weight",
        type=float,
        default=0.2,
        help="Minimum additive weight for each sample in foreground sampler (default 0.2)",
    )
    p.add_argument(
        "--dice_weight",
        type=float,
        default=0.4,
        help="Weight for Dice loss in total loss: CE + dice_weight*Dice (default 0.4)",
    )
    p.add_argument(
        "--fg_aux_weight",
        type=float,
        default=0.6,
        help="Weight for auxiliary foreground-vs-background loss (default 0.6)",
    )
    p.add_argument(
        "--fg_aux_warmup_epochs",
        type=int,
        default=20,
        help="Epochs to linearly warm up fg_aux_weight from 0 to target (default 20)",
    )
    p.add_argument(
        "--background_ce_weight",
        type=float,
        default=0.05,
        help="Maximum CE class weight for background class (default 0.05)",
    )
    p.add_argument(
        "--bg_ratio_weight",
        type=float,
        default=1.0,
        help="Weight for background-ratio matching loss (default 1.0)",
    )
    p.add_argument(
        "--bg_ratio_warmup_epochs",
        type=int,
        default=20,
        help="Epochs to linearly warm up bg_ratio_weight from 0 to target (default 20)",
    )
    p.add_argument(
        "--ignore_background_in_dice",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, exclude background class from Dice loss (default 1).",
    )
    p.add_argument(
        "--ignore_background_in_metrics",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, exclude background class from mIoU metric (default 1).",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for CrossEntropyLoss (default 0.05)",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss (default 2.0)",
    )
    p.add_argument(
        "--focal_class_weights",
        type=str,
        default="",
        help=(
            "Optional comma-separated per-class weights for focal loss, "
            "e.g. '1,3,1' for classes 0..C-1 (default none)."
        ),
    )
    p.add_argument(
        "--focal_focus_classes",
        type=str,
        default="all_fg",
        help=(
            "Classes used by focal pixel sampling. Use 'all_fg' (default), 'all', "
            "or comma-separated indices like '1,2,3,4,5,6,7,8'."
        ),
    )
    p.add_argument(
        "--class_weight_min",
        type=float,
        default=0.2,
        help="Lower clamp for estimated class weights (default 0.2)",
    )
    p.add_argument(
        "--class_weight_max",
        type=float,
        default=5.0,
        help="Upper clamp for estimated class weights (default 5.0)",
    )
    p.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience in epochs (default 5)",
    )
    p.add_argument(
        "--scheduler_monitor",
        type=str,
        default="val_miou",
        choices=["val_loss", "val_miou"],
        help="Metric for ReduceLROnPlateau: val_loss or val_miou (default val_miou)",
    )
    p.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau decay factor (default 0.5)",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for scheduler (default 1e-5)",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=30,
        help="Early stop after no val mIoU improvement for N epochs (0 disables, default 12)",
    )
    p.add_argument(
        "--save_val_samples",
        type=int,
        default=4,
        help="Number of validation samples to visualize and save (default 4)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split and reproducibility (default 42)",
    )
    args = p.parse_args(argv)

    # If YAML config provided, load and override matching arguments
    def _coerce_config_value(ref_value, new_value):
        if new_value is None:
            return new_value
        # Preserve bool before int (bool is subclass of int)
        if isinstance(ref_value, bool):
            if isinstance(new_value, str):
                v = new_value.strip().lower()
                if v in ("1", "true", "yes", "y", "on"):
                    return True
                if v in ("0", "false", "no", "n", "off"):
                    return False
            return bool(new_value)
        if isinstance(ref_value, int) and not isinstance(ref_value, bool):
            if isinstance(new_value, str):
                try:
                    return int(new_value)
                except ValueError:
                    return int(float(new_value))
            if isinstance(new_value, float):
                return int(new_value)
        if isinstance(ref_value, float):
            if isinstance(new_value, str):
                return float(new_value)
            return float(new_value)
        if isinstance(ref_value, str):
            return str(new_value)
        return new_value

    if getattr(args, "config", ""):
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"YAML config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                current = getattr(args, k)
                setattr(args, k, _coerce_config_value(current, v))
            else:
                print(f"Warning: unknown config key '{k}' in {cfg_path}, ignoring")

    missing = [
        name for name in ("embeddings_path", "labels_file", "output_dir")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Provide them via CLI or in the YAML config."
        )

    return args


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
