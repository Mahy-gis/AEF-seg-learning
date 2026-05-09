from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two MTS12 64D embedding datasets (AEF-provided tif vs self-generated npz), "
            "report statistics, and save paired visualizations."
        )
    )
    parser.add_argument(
        "--aef_root",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/MTS12/embeddings/AEF"),
        help="AEF embedding root directory (usually flat tif files like grid_123.tif).",
    )
    parser.add_argument(
        "--s12_root",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/MTS12/embeddings/S12"),
        help="Self-generated embedding root directory (contains train/val/test).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/result/mts12_embedding_compare"),
        help="Output directory for tables, summary, and plots.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to compare.",
    )
    parser.add_argument(
        "--bands",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Three embedding dimensions used as R/G/B for visualization.",
    )
    parser.add_argument(
        "--vis_count",
        type=int,
        default=8,
        help="Number of matched samples to visualize for each split.",
    )
    parser.add_argument(
        "--vis_mode",
        choices=["worst", "best", "random"],
        default="worst",
        help="How to select visualization samples based on MSE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when vis_mode=random.",
    )
    parser.add_argument(
        "--max_items_per_split",
        type=int,
        default=0,
        help="Optional limit of compared matched pairs per split. 0 means all.",
    )
    return parser.parse_args()


def list_npz_files(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*.npz"))
    return {p.name: p for p in files}


def list_tif_files(folder: Path) -> Dict[str, Path]:
    files = sorted(folder.glob("*.tif"))
    return {p.name: p for p in files}


def extract_numeric_id(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    if m is None:
        return None
    return int(m.group(1))


def build_aef_id_map(aef_root: Path, splits: Sequence[str]) -> Dict[int, Path]:
    id_map: Dict[int, Path] = {}

    # Priority 1: flat layout under root, e.g. grid_100.tif
    flat_tifs = sorted(aef_root.glob("*.tif"))
    for p in flat_tifs:
        sid = extract_numeric_id(p.name)
        if sid is not None:
            id_map[sid] = p

    # Priority 2: split layout, e.g. aef_root/train/*.tif
    if not id_map:
        for split in splits:
            split_dir = aef_root / split
            if not split_dir.exists():
                continue
            for p in sorted(split_dir.glob("*.tif")):
                sid = extract_numeric_id(p.name)
                if sid is not None:
                    id_map[sid] = p

    return id_map


def build_s12_records(s12_root: Path, splits: Sequence[str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    # Priority 1: split layout, e.g. s12_root/train/*.npz
    found_split = False
    for split in splits:
        split_dir = s12_root / split
        if not split_dir.exists():
            continue
        found_split = True
        for p in sorted(split_dir.glob("*.npz")):
            sid = extract_numeric_id(p.name)
            if sid is None:
                continue
            records.append({"split": split, "name": p.name, "path": p, "id": sid})

    # Priority 2: flat layout
    if not found_split:
        for p in sorted(s12_root.glob("*.npz")):
            sid = extract_numeric_id(p.name)
            if sid is None:
                continue
            records.append({"split": "all", "name": p.name, "path": p, "id": sid})

    return records


def load_npz_array(path: Path) -> np.ndarray:
    preferred_keys = [
        "embeddings",
        "embedding",
        "embeddings_native",
        "embeddings_per_time",
        "arr_0",
    ]
    with np.load(path, allow_pickle=True) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key])

        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                return np.asarray(arr)

    raise ValueError(f"No ndarray payload found in: {path}")


def to_hwc64(arr: np.ndarray, path: Path) -> np.ndarray:
    a = np.asarray(arr)

    if a.ndim == 3:
        if a.shape[-1] == 64:
            return a.astype(np.float32)
        if a.shape[0] == 64:
            return np.transpose(a, (1, 2, 0)).astype(np.float32)
        raise ValueError(f"Cannot locate 64D channel axis in shape={a.shape} from {path}")

    if a.ndim == 4:
        # Reduce temporal axis then convert to HWC if needed.
        if a.shape[-1] == 64:
            return a.mean(axis=0).astype(np.float32)
        if a.shape[1] == 64:
            return np.transpose(a.mean(axis=0), (1, 2, 0)).astype(np.float32)
        if a.shape[0] == 64:
            return np.transpose(a, (1, 2, 3, 0)).mean(axis=2).astype(np.float32)
        raise ValueError(f"Unsupported 4D embedding shape={a.shape} from {path}")

    raise ValueError(f"Unsupported ndim={a.ndim}, shape={a.shape} from {path}")


def load_tif_to_hwc64(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    # rasterio returns CHW; expected 64 channels.
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tif array, got shape={arr.shape} from {path}")
    if arr.shape[0] == 64:
        return np.transpose(arr, (1, 2, 0)).astype(np.float32)
    if arr.shape[-1] == 64:
        return arr.astype(np.float32)
    raise ValueError(f"Cannot locate 64 channels in tif shape={arr.shape} from {path}")


def align_spatial_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    c = min(a.shape[2], b.shape[2])
    return a[:h, :w, :c], b[:h, :w, :c]


def percentile_norm01(x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def get_rgb(arr_hwc64: np.ndarray, bands: Sequence[int]) -> np.ndarray:
    if len(bands) != 3:
        raise ValueError(f"bands must have 3 indices, got: {bands}")
    for b in bands:
        if b < 0 or b >= arr_hwc64.shape[-1]:
            raise ValueError(f"Band index {b} out of range for shape {arr_hwc64.shape}")
    rgb = arr_hwc64[..., [bands[0], bands[1], bands[2]]]
    return percentile_norm01(rgb)


def cosine_similarity_flat(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    xf = x.reshape(-1).astype(np.float64)
    yf = y.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(xf) * np.linalg.norm(yf) + eps
    return float(np.dot(xf, yf) / denom)


def compute_metrics(aef: np.ndarray, s12: np.ndarray) -> Dict[str, float]:
    diff = s12 - aef
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    mean_bias = float(np.mean(diff))
    cos = cosine_similarity_flat(aef, s12)

    aef_mean = float(np.mean(aef))
    s12_mean = float(np.mean(s12))
    aef_std = float(np.std(aef))
    s12_std = float(np.std(s12))
    std_ratio = float((s12_std + 1e-8) / (aef_std + 1e-8))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mean_bias_s12_minus_aef": mean_bias,
        "cosine_similarity": cos,
        "aef_mean": aef_mean,
        "s12_mean": s12_mean,
        "aef_std": aef_std,
        "s12_std": s12_std,
        "std_ratio_s12_over_aef": std_ratio,
    }


def save_pair_plot(
    out_path: Path,
    aef_rgb: np.ndarray,
    s12_rgb: np.ndarray,
    diff_rgb: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    axes[0].imshow(aef_rgb)
    axes[0].set_title("AEF")
    axes[0].axis("off")

    axes[1].imshow(s12_rgb)
    axes[1].set_title("S12")
    axes[1].axis("off")

    axes[2].imshow(diff_rgb)
    axes[2].set_title("|S12 - AEF|")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_numeric(df: pd.DataFrame, metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = df[m].to_numpy(dtype=np.float64)
        result[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }
    return result


def plot_channel_stats(
    out_path: Path,
    aef_channel_mean: np.ndarray,
    s12_channel_mean: np.ndarray,
    aef_channel_std: np.ndarray,
    s12_channel_std: np.ndarray,
) -> None:
    x = np.arange(aef_channel_mean.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140, sharex=True)

    axes[0].plot(x, aef_channel_mean, label="AEF mean", linewidth=1.5)
    axes[0].plot(x, s12_channel_mean, label="S12 mean", linewidth=1.5)
    axes[0].set_ylabel("mean")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, aef_channel_std, label="AEF std", linewidth=1.5)
    axes[1].plot(x, s12_channel_std, label="S12 std", linewidth=1.5)
    axes[1].set_xlabel("embedding channel index")
    axes[1].set_ylabel("std")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def choose_visualization_rows(df: pd.DataFrame, vis_count: int, vis_mode: str, seed: int) -> pd.DataFrame:
    if df.empty or vis_count <= 0:
        return df.iloc[:0]

    vis_count = min(vis_count, len(df))
    if vis_mode == "worst":
        return df.sort_values("mse", ascending=False).head(vis_count)
    if vis_mode == "best":
        return df.sort_values("mse", ascending=True).head(vis_count)
    return df.sample(n=vis_count, random_state=seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    for band in args.bands:
        if band < 0 or band >= 64:
            raise ValueError(f"Invalid band index {band}; must be in [0, 63].")

    ensure_dir(args.out_dir)

    aef_id_map = build_aef_id_map(args.aef_root, args.splits)
    if not aef_id_map:
        raise FileNotFoundError(f"No tif embeddings found under AEF root: {args.aef_root}")

    s12_records = build_s12_records(args.s12_root, args.splits)
    if not s12_records:
        raise FileNotFoundError(f"No npz embeddings found under S12 root: {args.s12_root}")

    print(f"AEF ids found: {len(aef_id_map)}")
    print(f"S12 files found: {len(s12_records)}")

    all_rows: List[Dict[str, object]] = []
    split_summaries: Dict[str, Dict[str, object]] = {}

    for split in args.splits:
        split_records = [r for r in s12_records if str(r["split"]) == split]
        if not split_records:
            print(f"[WARN] Skip split={split}, no S12 files found.")
            continue

        split_s12_ids = {int(r["id"]) for r in split_records}
        matched_records = [r for r in split_records if int(r["id"]) in aef_id_map]

        if args.max_items_per_split > 0:
            matched_records = matched_records[: args.max_items_per_split]

        matched_ids = {int(r["id"]) for r in matched_records}
        only_s12 = len(split_s12_ids - set(aef_id_map.keys()))
        only_aef = len(set(aef_id_map.keys()) - split_s12_ids)

        print(f"split={split}: matched={len(matched_records)}, only_aef={only_aef}, only_s12={only_s12}")
        if not matched_records:
            continue

        split_rows: List[Dict[str, object]] = []
        # Channel-wise streaming stats for per-dimension drift insight.
        ch_sum_aef = np.zeros(64, dtype=np.float64)
        ch_sum_s12 = np.zeros(64, dtype=np.float64)
        ch_sqsum_aef = np.zeros(64, dtype=np.float64)
        ch_sqsum_s12 = np.zeros(64, dtype=np.float64)
        ch_count = 0

        for rec in tqdm(matched_records, desc=f"compare-{split}"):
            sid = int(rec["id"])
            name = str(rec["name"])
            aef_path = aef_id_map[sid]
            s12_path = Path(rec["path"])
            try:
                aef = load_tif_to_hwc64(aef_path)
                s12 = to_hwc64(load_npz_array(s12_path), s12_path)
                aef, s12 = align_spatial_shape(aef, s12)
                if aef.shape[-1] != 64 or s12.shape[-1] != 64:
                    raise ValueError(f"Expected 64 channels, got {aef.shape} and {s12.shape}")

                metrics = compute_metrics(aef, s12)
                row: Dict[str, object] = {
                    "split": split,
                    "name": name,
                    "id": sid,
                    "aef_path": str(aef_path),
                    "s12_path": str(s12_path),
                    "height": int(aef.shape[0]),
                    "width": int(aef.shape[1]),
                }
                row.update(metrics)
                split_rows.append(row)

                aef_flat = aef.reshape(-1, 64)
                s12_flat = s12.reshape(-1, 64)
                ch_sum_aef += aef_flat.sum(axis=0)
                ch_sum_s12 += s12_flat.sum(axis=0)
                ch_sqsum_aef += np.square(aef_flat).sum(axis=0)
                ch_sqsum_s12 += np.square(s12_flat).sum(axis=0)
                ch_count += aef_flat.shape[0]
            except Exception as exc:
                print(f"[WARN] split={split}, name={name}, error={exc}")

        if not split_rows:
            print(f"[WARN] split={split} has no valid matched sample after loading.")
            continue

        split_df = pd.DataFrame(split_rows)
        all_rows.extend(split_rows)

        split_out = args.out_dir / split
        vis_out = split_out / "visualizations"
        ensure_dir(split_out)
        ensure_dir(vis_out)

        split_csv = split_out / "sample_metrics.csv"
        split_df.to_csv(split_csv, index=False)

        metrics_to_summarize = [
            "mse",
            "mae",
            "rmse",
            "cosine_similarity",
            "mean_bias_s12_minus_aef",
            "std_ratio_s12_over_aef",
        ]
        split_summary = {
            "split": split,
            "matched": int(len(matched_records)),
            "matched_ids": int(len(matched_ids)),
            "valid_compared": int(len(split_df)),
            "only_aef": int(only_aef),
            "only_s12": int(only_s12),
            "metrics": summarize_numeric(split_df, metrics_to_summarize),
        }

        aef_channel_mean = ch_sum_aef / max(ch_count, 1)
        s12_channel_mean = ch_sum_s12 / max(ch_count, 1)
        aef_channel_var = ch_sqsum_aef / max(ch_count, 1) - np.square(aef_channel_mean)
        s12_channel_var = ch_sqsum_s12 / max(ch_count, 1) - np.square(s12_channel_mean)
        aef_channel_std = np.sqrt(np.clip(aef_channel_var, 0.0, None))
        s12_channel_std = np.sqrt(np.clip(s12_channel_var, 0.0, None))

        channel_table = pd.DataFrame(
            {
                "channel": np.arange(64),
                "aef_mean": aef_channel_mean,
                "s12_mean": s12_channel_mean,
                "mean_diff_s12_minus_aef": s12_channel_mean - aef_channel_mean,
                "aef_std": aef_channel_std,
                "s12_std": s12_channel_std,
                "std_diff_s12_minus_aef": s12_channel_std - aef_channel_std,
            }
        )
        channel_table.to_csv(split_out / "channel_stats.csv", index=False)
        plot_channel_stats(
            split_out / "channel_stats.png",
            aef_channel_mean=aef_channel_mean,
            s12_channel_mean=s12_channel_mean,
            aef_channel_std=aef_channel_std,
            s12_channel_std=s12_channel_std,
        )

        vis_rows = choose_visualization_rows(split_df, args.vis_count, args.vis_mode, args.seed)
        for _, r in vis_rows.iterrows():
            name = str(r["name"])
            aef_path = Path(str(r["aef_path"]))
            s12_path = Path(str(r["s12_path"]))
            aef = load_tif_to_hwc64(aef_path)
            s12 = to_hwc64(load_npz_array(s12_path), s12_path)
            aef, s12 = align_spatial_shape(aef, s12)

            aef_rgb = get_rgb(aef, args.bands)
            s12_rgb = get_rgb(s12, args.bands)
            diff_rgb = percentile_norm01(np.abs(s12_rgb - aef_rgb), low=0.0, high=99.0)

            title = (
                f"{split}/{name} | bands={args.bands} | "
                f"MSE={r['mse']:.5f}, MAE={r['mae']:.5f}, cos={r['cosine_similarity']:.5f}"
            )
            out_png = vis_out / f"{Path(name).stem}_pair.png"
            save_pair_plot(out_png, aef_rgb, s12_rgb, diff_rgb, title=title)

        with (split_out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(split_summary, f, ensure_ascii=False, indent=2)

        split_summaries[split] = split_summary

    if not all_rows:
        print("No valid matched samples compared. Please check dataset structure and npz keys.")
        return

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(args.out_dir / "all_sample_metrics.csv", index=False)

    overall_metrics = [
        "mse",
        "mae",
        "rmse",
        "cosine_similarity",
        "mean_bias_s12_minus_aef",
        "std_ratio_s12_over_aef",
    ]
    overall_summary = {
        "bands_for_visualization": list(args.bands),
        "total_valid_samples": int(len(all_df)),
        "overall_metrics": summarize_numeric(all_df, overall_metrics),
        "split_summaries": split_summaries,
    }
    with (args.out_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print("Comparison completed.")
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
