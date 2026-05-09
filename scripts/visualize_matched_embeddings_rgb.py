from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match AEF/S12 embeddings by id and save RGB visualizations."
    )
    parser.add_argument(
        "--aef_dir",
        type=Path,
        default=Path(
            "/mnt/data/mhy/RSFM/AEF-seg-learning/data/MTS12/embeddings/AEF_embeddings/train"
        ),
    )
    parser.add_argument(
        "--s12_dir",
        type=Path,
        default=Path(
            "/mnt/data/mhy/RSFM/AEF-seg-learning/data/MTS12/embeddings/S12/train"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "/mnt/data/mhy/RSFM/AEF-seg-learning/data/MTS12/embeddings/vis_A01_A16_A09"
        ),
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=3,
        default=[2, 17, 10],
        help="Embedding dimensions for RGB in order (R, G, B).",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Maximum matched pairs to visualize; 0 means all.",
    )
    return parser.parse_args()


def build_id_map(files: Iterable[Path], pattern: str) -> Dict[int, Path]:
    id_map: Dict[int, Path] = {}
    regex = re.compile(pattern)
    for path in files:
        match = regex.search(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        id_map[idx] = path
    return id_map


def load_embedding_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
        return np.asarray(arr)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "embeddings" in data:
                return np.asarray(data["embeddings"])
            if "embeddings_native" in data:
                return np.asarray(data["embeddings_native"])
            if "embeddings_per_time" in data:
                return np.asarray(data["embeddings_per_time"])

            keys = list(data.files)
            for k in keys:
                v = data[k]
                if isinstance(v, np.ndarray):
                    return np.asarray(v)
            raise ValueError(f"No ndarray payload found in {path}")

    raise ValueError(f"Unsupported file type: {path}")


def to_hwc64(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)

    if a.ndim == 3:
        if a.shape[-1] == 64:
            return a
        if a.shape[0] == 64:
            return np.transpose(a, (1, 2, 0))
        raise ValueError(f"3D embedding does not contain 64-channel axis: shape={a.shape}")

    if a.ndim == 4:
        if a.shape[-1] == 64:
            # (T,H,W,64): average over time for visualization.
            return a.mean(axis=0)
        if a.shape[1] == 64:
            # (T,64,H,W): average over time and convert to HWC.
            return np.transpose(a.mean(axis=0), (1, 2, 0))
        raise ValueError(f"4D embedding does not contain 64-channel axis: shape={a.shape}")

    raise ValueError(f"Unsupported embedding ndim={a.ndim}, shape={a.shape}")


def percentile_norm01(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.nanpercentile(x, p_low)
    hi = np.nanpercentile(x, p_high)
    if hi <= lo:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def rgb_from_dims(arr_hwc64: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
    if arr_hwc64.shape[-1] < 64:
        raise ValueError(f"Expected >=64 channels, got {arr_hwc64.shape}")
    rgb = arr_hwc64[..., [dims[0], dims[1], dims[2]]]
    return percentile_norm01(rgb)


def save_pair_figure(
    out_path: Path,
    rgb_aef: np.ndarray,
    rgb_s12: np.ndarray,
    sample_id: int,
    aef_name: str,
    s12_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].imshow(rgb_aef)
    axes[0].set_title(f"AEF id={sample_id}\n{aef_name}")
    axes[0].axis("off")

    axes[1].imshow(rgb_s12)
    axes[1].set_title(f"S12 id={sample_id}\n{s12_name}")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.aef_dir.exists():
        raise FileNotFoundError(f"AEF dir not found: {args.aef_dir}")
    if not args.s12_dir.exists():
        raise FileNotFoundError(f"S12 dir not found: {args.s12_dir}")

    out_dir = args.out_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = tuple(args.dims)
    for d in dims:
        if d < 0 or d >= 64:
            raise ValueError(f"Dimension index out of range [0, 63]: {d}")

    aef_files = sorted(args.aef_dir.glob("*.npy"))
    s12_files = sorted(args.s12_dir.glob("*.npz"))

    aef_map = build_id_map(aef_files, r"eopath_(\d+)_")
    s12_map = build_id_map(s12_files, r"embedding_(\d+)\.npz$")

    common_ids = sorted(set(aef_map.keys()) & set(s12_map.keys()))
    if args.max_items > 0:
        common_ids = common_ids[: args.max_items]

    if not common_ids:
        print("No matched ids found.")
        return

    success = 0
    for sid in common_ids:
        aef_path = aef_map[sid]
        s12_path = s12_map[sid]
        try:
            aef_arr = to_hwc64(load_embedding_array(aef_path))
            s12_arr = to_hwc64(load_embedding_array(s12_path))

            rgb_aef = rgb_from_dims(aef_arr, dims)
            rgb_s12 = rgb_from_dims(s12_arr, dims)

            out_path = out_dir / f"id_{sid:05d}_A01A16A09_pair.png"
            save_pair_figure(
                out_path=out_path,
                rgb_aef=rgb_aef,
                rgb_s12=rgb_s12,
                sample_id=sid,
                aef_name=aef_path.name,
                s12_name=s12_path.name,
            )
            success += 1
        except Exception as exc:
            print(f"[WARN] skip id={sid}: {exc}")

    print(f"Matched ids: {len(common_ids)}")
    print(f"Saved visualizations: {success}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
