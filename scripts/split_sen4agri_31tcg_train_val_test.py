#!/usr/bin/env python3
"""Randomly split Sen4Agri_31TCG image/label pairs into train/val/test.

- Expects source files:
  - image: *_image.npy
  - label: *_label.npy
- Pairs are matched by the prefix before the final "_image.npy"/"_label.npy".

Default destinations follow AEF-seg-learning convention:
  data/Sen4Agri_31TCG/raw/{train,val,test}
  data/Sen4Agri_31TCG/label/{train,val,test}

Also writes a JSON manifest under: data/Sen4Agri_31TCG/split.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_SUFFIX = "_image.npy"
LABEL_SUFFIX = "_label.npy"


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int


def _stem_from_name(name: str, suffix: str) -> str | None:
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _collect_pairs(image_dir: Path, label_dir: Path) -> List[Tuple[str, Path, Path]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    images: Dict[str, Path] = {}
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        stem = _stem_from_name(p.name, IMAGE_SUFFIX)
        if stem is None:
            continue
        images[stem] = p

    labels: Dict[str, Path] = {}
    for p in label_dir.iterdir():
        if not p.is_file():
            continue
        stem = _stem_from_name(p.name, LABEL_SUFFIX)
        if stem is None:
            continue
        labels[stem] = p

    common = sorted(set(images.keys()) & set(labels.keys()))
    missing_labels = sorted(set(images.keys()) - set(labels.keys()))
    missing_images = sorted(set(labels.keys()) - set(images.keys()))

    if missing_labels or missing_images:
        msg = ["Unpaired files detected:"]
        if missing_labels:
            msg.append(f"  images without labels: {len(missing_labels)} (e.g. {missing_labels[:5]})")
        if missing_images:
            msg.append(f"  labels without images: {len(missing_images)} (e.g. {missing_images[:5]})")
        raise RuntimeError("\n".join(msg))

    return [(stem, images[stem], labels[stem]) for stem in common]


def _compute_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> SplitCounts:
    if n <= 0:
        return SplitCounts(train=0, val=0, test=0)

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of ratios must be > 0")

    train = int(math.floor(n * (train_ratio / total)))
    val = int(math.floor(n * (val_ratio / total)))
    test = n - train - val

    # Ensure all splits non-negative and sum matches.
    if train < 0 or val < 0 or test < 0 or (train + val + test) != n:
        raise AssertionError("Invalid split counts")

    return SplitCounts(train=train, val=val, test=test)


def _ensure_dirs(root_raw: Path, root_label: Path) -> None:
    for split in ("train", "val", "test"):
        (root_raw / split).mkdir(parents=True, exist_ok=True)
        (root_label / split).mkdir(parents=True, exist_ok=True)


def _copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path, *, overwrite: bool) -> None:
    if (img_dst.exists() or lbl_dst.exists()) and not overwrite:
        raise FileExistsError(f"Destination exists (use --overwrite): {img_dst} / {lbl_dst}")
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-image-dir",
        type=Path,
        default=Path("/mnt/data/mhy/dataset/Sen4Agri_31TCG/image"),
        help="Source directory containing *_image.npy",
    )
    parser.add_argument(
        "--src-label-dir",
        type=Path,
        default=Path("/mnt/data/mhy/dataset/Sen4Agri_31TCG/label"),
        help="Source directory containing *_label.npy",
    )
    parser.add_argument(
        "--dst-raw-root",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/raw"),
        help="Destination root for images (will create train/val/test)",
    )
    parser.add_argument(
        "--dst-label-root",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/label"),
        help="Destination root for labels (will create train/val/test)",
    )
    parser.add_argument("--train-ratio", type=float, default=7.0)
    parser.add_argument("--val-ratio", type=float, default=1.5)
    parser.add_argument("--test-ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/split.json"),
        help="Write JSON manifest of stems for each split",
    )

    args = parser.parse_args()

    pairs = _collect_pairs(args.src_image_dir, args.src_label_dir)
    n = len(pairs)
    counts = _compute_counts(n, args.train_ratio, args.val_ratio, args.test_ratio)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    splits: Dict[str, List[str]] = {
        "train": [stem for stem, _, _ in pairs[: counts.train]],
        "val": [stem for stem, _, _ in pairs[counts.train : counts.train + counts.val]],
        "test": [stem for stem, _, _ in pairs[counts.train + counts.val :]],
    }

    _ensure_dirs(args.dst_raw_root, args.dst_label_root)

    stem_to_paths = {stem: (img, lbl) for stem, img, lbl in pairs}
    for split_name, stems in splits.items():
        for stem in stems:
            img_src, lbl_src = stem_to_paths[stem]
            img_dst = args.dst_raw_root / split_name / img_src.name
            lbl_dst = args.dst_label_root / split_name / lbl_src.name
            _copy_pair(img_src, lbl_src, img_dst, lbl_dst, overwrite=args.overwrite)

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {"train": counts.train, "val": counts.val, "test": counts.test},
        "splits": splits,
    }
    args.manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Done. Total={n} | train={counts.train} val={counts.val} test={counts.test}\n"
        f"Images -> {args.dst_raw_root}\n"
        f"Labels -> {args.dst_label_root}\n"
        f"Manifest -> {args.manifest_path}"
    )


if __name__ == "__main__":
    main()
