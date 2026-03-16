from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GEEMultiSourceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.normalize = normalize

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.files = sorted(self.data_dir.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {self.data_dir}")

        # 自动检测实际存在的数据源，兼容仅 S1/S2 或 L8/S1/S2 等多种组合
        detected_sources: List[str] = []
        for f in self.files:
            try:
                with np.load(f) as data:
                    keys = set(data.keys())
                    candidates = ["landsat", "sentinel2", "sentinel1"]
                    detected_sources = [k for k in candidates if k in keys]
                    if detected_sources:
                        break
            except Exception:
                continue

        if not detected_sources:
            raise RuntimeError(
                "No known sources (landsat/sentinel1/sentinel2) found in any sample_*.npz; "
                "please check download_gee_l8_s1_s2 outputs."
            )

        self.sources = detected_sources

        # 过滤在所有已检测源上均为全 0 的样本
        valid_files: List[Path] = []
        for f in self.files:
            try:
                with np.load(f) as data:
                    has_any_signal = False
                    for src in self.sources:
                        arr = data[src]
                        if np.any(arr != 0):
                            has_any_signal = True
                            break
                    if has_any_signal:
                        valid_files.append(f)
            except Exception:
                # 如果单个文件损坏或缺键，则直接跳过
                continue

        if not valid_files:
            raise RuntimeError(
                "All GEE multi-source samples appear to be all-zero across available sources; "
                "please check download_gee_l8_s1_s2 outputs."
            )

        self.files = valid_files

    def __len__(self) -> int:
        return len(self.files)

    def _normalize(self, arr: np.ndarray, source: Optional[str] = None) -> np.ndarray:
        if not self.normalize:
            return arr
        arr = arr.astype(np.float32)
        if arr.ndim == 4:
            normalized = np.zeros_like(arr, dtype=np.float32)
            for channel_idx in range(arr.shape[-1]):
                band = arr[..., channel_idx]

                # For optical sources (especially S2), many padded/missing pixels are 0.
                # Use robust percentiles over non-zero pixels to avoid dynamic range collapse.
                if source in {"sentinel2", "landsat"}:
                    valid = ~np.isclose(band, 0.0)
                    if np.any(valid):
                        vals = band[valid]
                        v_min = np.nanpercentile(vals, 2)
                        v_max = np.nanpercentile(vals, 98)
                        if v_max > v_min:
                            band_norm = (band - v_min) / (v_max - v_min)
                            band_norm = np.clip(band_norm, 0.0, 1.0)
                            # Keep explicit invalid pixels as 0
                            band_norm[~valid] = 0.0
                            normalized[..., channel_idx] = band_norm
                        else:
                            normalized[..., channel_idx] = band
                    else:
                        normalized[..., channel_idx] = band
                else:
                    v_min = np.nanmin(band)
                    v_max = np.nanmax(band)
                    if v_max > v_min:
                        normalized[..., channel_idx] = (band - v_min) / (v_max - v_min)
                    else:
                        normalized[..., channel_idx] = band
            return normalized

        v_min = np.nanmin(arr)
        v_max = np.nanmax(arr)
        if v_max > v_min:
            return (arr - v_min) / (v_max - v_min)
        return arr

    def _frame_valid_mask(self, arr: np.ndarray) -> np.ndarray:
        return ~np.all(np.isclose(arr, 0.0), axis=(1, 2, 3))

    def _pad_time(self, x: np.ndarray, valid_mask: np.ndarray, target_T: int) -> tuple[np.ndarray, np.ndarray]:
        t, h, w, c = x.shape
        if t == target_T:
            return x, valid_mask

        if valid_mask.any():
            pad_frame = x[np.flatnonzero(valid_mask)[-1]:np.flatnonzero(valid_mask)[-1] + 1]
        else:
            pad_frame = x[-1:]

        pad = np.repeat(pad_frame, target_T - t, axis=0)
        pad_mask = np.zeros((target_T - t,), dtype=bool)
        return np.concatenate([x, pad], axis=0), np.concatenate([valid_mask, pad_mask], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np.load(path) as data:
            timestamps = data["timestamps"]  # (T,)

            arrays: Dict[str, np.ndarray] = {}
            frame_valid: Dict[str, np.ndarray] = {}

            T_list: List[int] = []
            H = W = None
            for src in self.sources:
                arr = data[src]
                if arr.ndim != 4:
                    raise ValueError(f"Source {src} must have shape (T, H, W, C), got {arr.shape}")
                T_src, H_src, W_src, _ = arr.shape
                if H_src != self.patch_size or W_src != self.patch_size:
                    raise ValueError(
                        "All inputs must share the same patch size and match patch_size; "
                        "adjust download script or dataset."
                    )
                arrays[src] = arr
                frame_valid[src] = self._frame_valid_mask(arr)
                T_list.append(T_src)
                H, W = H_src, W_src

        T = int(max(T_list))

        # 对每个源在时间维进行 pad，并做归一化
        for src in self.sources:
            arr, mask = self._pad_time(arrays[src], frame_valid[src], T)
            arrays[src] = self._normalize(arr, source=src)
            frame_valid[src] = mask

        # 至少在任一源上为非 0 的时间步才保留
        keep_mask = np.zeros((T,), dtype=bool)
        for src in self.sources:
            keep_mask |= frame_valid[src]

        ts = np.array(timestamps, dtype=np.float64)
        if ts.shape[0] < T:
            if ts.shape[0] == 0:
                base = 0.0
            else:
                base = float(ts[-1])
            pad_ts = np.full((T - ts.shape[0],), base, dtype=np.float64)
            ts = np.concatenate([ts, pad_ts], axis=0)

        # 若存在需要过滤的时间步且不全部被过滤，则按掩码裁剪时间维
        if keep_mask.any() and not keep_mask.all():
            ts = ts[keep_mask]
            for src in self.sources:
                arrays[src] = arrays[src][keep_mask]
                frame_valid[src] = frame_valid[src][keep_mask]

        ts_tensor = torch.from_numpy(ts.astype(np.float32))

        source_data_tensors: Dict[str, torch.Tensor] = {}
        frame_mask_tensors: Dict[str, torch.Tensor] = {}
        for src in self.sources:
            source_data_tensors[src] = torch.from_numpy(arrays[src]).float()
            frame_mask_tensors[src] = torch.from_numpy(frame_valid[src].astype(np.bool_))

        if ts.size > 0:
            valid_start = float(ts[0])
            valid_end = float(ts[-1])
        else:
            valid_start = 0.0
            valid_end = 0.0

        return {
            "source_data": source_data_tensors,
            "timestamps": {src: ts_tensor for src in self.sources},
            "frame_valid_mask": frame_mask_tensors,
            "valid_period": (valid_start, valid_end),
        }


def create_gee_multisource_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = 256,
    normalize: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GEEMultiSourceDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        normalize=normalize,
    )

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_names = list(batch[0]["source_data"].keys())
        B = len(batch)

        collated_sources: Dict[str, torch.Tensor] = {}
        collated_timestamps: Dict[str, torch.Tensor] = {}
        collated_frame_masks: Dict[str, torch.Tensor] = {}

        for src in source_names:
            tensors = [sample["source_data"][src] for sample in batch]
            masks = [sample["frame_valid_mask"][src] for sample in batch]
            T_max = max(t.shape[0] for t in tensors)
            padded: List[torch.Tensor] = []
            padded_masks: List[torch.Tensor] = []
            for t, mask in zip(tensors, masks):
                if t.shape[0] < T_max:
                    T, H, W, C = t.shape
                    pad_frame = t[-1:].repeat(T_max - T, 1, 1, 1)
                    pad = pad_frame if T > 0 else torch.zeros(T_max - T, H, W, C, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=0)
                    mask = torch.cat([mask, torch.zeros(T_max - mask.shape[0], dtype=torch.bool)], dim=0)
                padded.append(t)
                padded_masks.append(mask)
            collated_sources[src] = torch.stack(padded)
            collated_frame_masks[src] = torch.stack(padded_masks)

            ts_list = [sample["timestamps"][src] for sample in batch]
            T_ts_max = max(len(ts) for ts in ts_list)
            ts_padded: List[torch.Tensor] = []
            for ts in ts_list:
                if ts.shape[0] < T_ts_max:
                    last = ts[-1] if ts.shape[0] > 0 else torch.tensor(0.0, dtype=ts.dtype)
                    pad_ts = torch.full((T_ts_max - ts.shape[0],), float(last), dtype=ts.dtype)
                    ts = torch.cat([ts, pad_ts])
                ts_padded.append(ts)
            collated_timestamps[src] = torch.stack(ts_padded)

        valid_periods = [sample["valid_period"] for sample in batch]

        return {
            "source_data": collated_sources,
            "timestamps": collated_timestamps,
            "frame_valid_mask": collated_frame_masks,
            "valid_periods": valid_periods,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
