import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from seg.train_unet_from_embeddings import labels_to_rgb, load_label_mask_from_file, prepare_features_from_embeddings
except ModuleNotFoundError:
    from train_unet_from_embeddings import labels_to_rgb, load_label_mask_from_file, prepare_features_from_embeddings


def _extract_id(path: Path) -> Optional[str]:
    name = path.name
    stem = path.stem

    if stem.endswith("_image"):
        return stem[: -len("_image")]
    if stem.endswith("_label"):
        return stem[: -len("_label")]

    if stem.startswith("embedding_"):
        token = stem[len("embedding_") :]
        if token.isdigit():
            return str(int(token))
        return token

    if stem.startswith("sample_"):
        token = stem[len("sample_") :]
        if token.isdigit():
            return str(int(token))
        return token

    if stem.startswith("ParcelIDs_"):
        match = re.match(r"^ParcelIDs_(\d+)", stem)
        if match:
            return str(int(match.group(1)))

    if stem.isdigit():
        return str(int(stem))

    return stem or None


def _collect_files(root: Path, split: str, patterns: Sequence[str]) -> Dict[str, Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        split_dir = root

    files: Dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(split_dir.glob(pattern)):
            if not path.is_file():
                continue
            sample_id = _extract_id(path)
            if sample_id is None:
                continue
            files[sample_id] = path
    return files


def _load_raw_sentinel2(raw_file: Path, s2_key: str = "sentinel2") -> np.ndarray:
    payload = np.load(raw_file, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        container = {k: payload[k] for k in payload.files}
        payload.close()
    elif isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        container = payload.item()
    else:
        container = payload

    if isinstance(container, dict):
        if s2_key in container:
            s2 = container[s2_key]
        elif "S2" in container:
            s2 = container["S2"]
        elif "sentinel2" in container:
            s2 = container["sentinel2"]
        else:
            available = list(container.keys())
            raise KeyError(f"Sentinel-2 key not found in {raw_file}. Available keys: {available}")
    else:
        s2 = container

    s2 = np.asarray(s2)
    if s2.ndim == 4:
        if s2.shape[-1] <= 64:
            s2 = s2[-1]
        elif s2.shape[1] <= 64:
            s2 = s2[-1].transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported Sentinel-2 array shape in {raw_file}: {s2.shape}")

    if s2.ndim != 3:
        raise ValueError(f"Expected 3D Sentinel-2 tensor after selection in {raw_file}, got {s2.shape}")

    if s2.shape[0] <= 64 and s2.shape[-1] > 64:
        chw = s2.astype(np.float32, copy=False)
    elif s2.shape[-1] <= 64:
        chw = np.transpose(s2, (2, 0, 1)).astype(np.float32, copy=False)
    else:
        chw = s2.astype(np.float32, copy=False)
    return chw


def _to_display_rgb(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"Expected (C,H,W) features, got {features.shape}")

    if features.shape[0] >= 3:
        rgb = features[:3]
    else:
        rgb = np.repeat(features[:1], 3, axis=0)

    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32, copy=False)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    if hi > lo:
        rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    else:
        rgb = rgb - rgb.min()
        rgb = rgb / (rgb.max() - rgb.min() + 1e-6)
    return rgb


def _format_id(sample_id: str) -> str:
    if sample_id.isdigit():
        return f"{int(sample_id):05d}"
    return sample_id


def _parse_id_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            out.append(str(int(token)))
        else:
            out.append(token)
    return out


def _save_triplet_figure(
    sample_id: str,
    aef_file: Path,
    s12_file: Optional[Path],
    raw_file: Path,
    label_file: Path,
    output_dir: Path,
    annotation: str,
    embedding_key: str,
    label_key: str,
    ignore_index: int,
) -> None:
    aef = prepare_features_from_embeddings(aef_file, embedding_key=embedding_key)
    s12 = None
    if s12_file is not None:
        s12 = prepare_features_from_embeddings(s12_file, embedding_key=embedding_key)
    raw_s2 = _load_raw_sentinel2(raw_file)
    label = load_label_mask_from_file(label_file, label_key=label_key)

    if label.shape != aef.shape[-2:]:
        try:
            from seg.train_unet_from_embeddings import resize_labels_to
        except ModuleNotFoundError:
            from train_unet_from_embeddings import resize_labels_to
        label = resize_labels_to(aef, label)

    aef_rgb = _to_display_rgb(aef)
    raw_rgb = _to_display_rgb(raw_s2)
    label_max = int(np.max(label)) if label.size else 0
    num_classes = max(label_max + 1, ignore_index + 1)
    label_rgb = labels_to_rgb(
        label.astype(np.int64),
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    panels = [
        (aef_rgb, "AEF embedding", "first 3 channels"),
    ]
    if s12 is not None:
        s12_rgb = _to_display_rgb(s12)
        panels.append((s12_rgb, "S12 embedding", "first 3 channels"))
    panels.extend(
        [
            (raw_rgb, "Sentinel-2 raw", "last timestep, first 3 channels"),
            (label_rgb, "Label", "segmentation mask"),
        ]
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (image, title, subtitle) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(f"{title}\n{subtitle}")
        ax.axis("off")

    fig.suptitle(f"ID {_format_id(sample_id)} | {annotation}", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{annotation}_{_format_id(sample_id)}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize matching samples from AEF embeddings, S12 embeddings, "
            "raw Sentinel-2 data, and labels using the same ID."
        )
    )
    parser.add_argument("--aef_root", type=str, required=True, help="Root folder of AEF embeddings")
    parser.add_argument(
        "--s12_emb_root",
        type=str,
        default="",
        help="Optional root folder of S12 embeddings",
    )
    parser.add_argument("--raw_root", type=str, required=True, help="Root folder of raw Sentinel-2 samples")
    parser.add_argument(
        "--label_root",
        type=str,
        default="",
        help="Optional label root; defaults to --raw_root",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--annotation", type=str, default="AEF-S12-raw-label")
    parser.add_argument("--embedding_key", type=str, default="auto")
    parser.add_argument("--label_key", type=str, default="labels")
    parser.add_argument("--ignore_index", type=int, default=19)
    parser.add_argument("--ids", type=str, default="", help="Optional comma-separated list of IDs to render")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of IDs to render; 0 means all")
    parser.add_argument(
        "--random_samples",
        action="store_true",
        help="Randomly sample max_samples IDs instead of taking the first ones",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    aef_root = Path(args.aef_root)
    s12_emb_root = Path(args.s12_emb_root) if args.s12_emb_root else None
    raw_root = Path(args.raw_root)
    label_root = Path(args.label_root) if args.label_root else raw_root
    output_dir = Path(args.output_dir) / args.annotation / args.split

    aef_files = _collect_files(aef_root, args.split, ("embedding_*.npy", "embedding_*.npz"))
    s12_files: Dict[str, Path] = {}
    if s12_emb_root is not None:
        s12_files = _collect_files(s12_emb_root, args.split, ("embedding_*.npy", "embedding_*.npz"))
    raw_files = _collect_files(
        raw_root,
        args.split,
        ("sample_*.npy", "sample_*.npz", "*_image.npy"),
    )

    label_split_dir = label_root / f"{args.split}_labels_npz"
    if not label_split_dir.is_dir():
        label_split_dir = label_root / f"{args.split}_labels"
    if not label_split_dir.is_dir():
        label_split_dir = label_root / args.split
    if not label_split_dir.is_dir():
        label_split_dir = label_root

    label_files = _collect_files(
        label_split_dir,
        "",
        ("ParcelIDs_*_labels.npy", "ParcelIDs_*_labels.npz", "*_label.npy"),
    )

    required_sets = [set(aef_files), set(raw_files), set(label_files)]
    if s12_files:
        required_sets.append(set(s12_files))
    common_ids = sorted(set.intersection(*required_sets))
    requested_ids = _parse_id_list(args.ids)
    if requested_ids:
        requested_set = set(requested_ids)
        common_ids = [sample_id for sample_id in common_ids if sample_id in requested_set]

    if args.random_samples and args.max_samples > 0:
        rng = np.random.RandomState(args.seed)
        common_ids = list(rng.choice(common_ids, size=min(args.max_samples, len(common_ids)), replace=False))
    elif args.max_samples > 0:
        common_ids = common_ids[: args.max_samples]

    if not common_ids:
        raise RuntimeError(
            f"No matched IDs found for split='{args.split}'. Checked AEF, S12, raw, and label directories."
        )

    print(f"Found {len(common_ids)} matched IDs for split '{args.split}'. Saving to {output_dir}")
    for sample_id in common_ids:
        _save_triplet_figure(
            sample_id=sample_id,
            aef_file=aef_files[sample_id],
            s12_file=s12_files.get(sample_id),
            raw_file=raw_files[sample_id],
            label_file=label_files[sample_id],
            output_dir=output_dir,
            annotation=args.annotation,
            embedding_key=args.embedding_key,
            label_key=args.label_key,
            ignore_index=args.ignore_index,
        )
        print(f"Saved {args.annotation}_{_format_id(sample_id)}.png")


if __name__ == "__main__":
    main()