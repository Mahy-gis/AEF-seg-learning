#!/usr/bin/env python3
"""Remap Sen4Agri_31TCG labels to 11 classes and write them by split.

This script reads the existing split manifest at:
  data/Sen4Agri_31TCG/split.json

It consumes the flat source labels from:
  /mnt/data/mhy/dataset/Sen4Agri_31TCG/label

And replaces the destination label tree at:
  /mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/label

The destination is recreated as:
  label/train
  label/val
  label/test

All labels are remapped with ignore index 255.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm


MAPPING: Dict[int, int] = {
    0: 0,
    110: 1,
    120: 2,
    140: 3,
    150: 4,
    160: 5,
    170: 6,
    330: 7,
    361: 7,
    435: 8,
    442: 8,
    438: 9,
    510: 10,
    770: 11,
}
IGNORE_INDEX = 255


def load_label(path: Path) -> np.ndarray:
    label = np.load(path, allow_pickle=True)
    if isinstance(label, np.ndarray) and label.dtype == object and label.shape == ():
        label = label.item()
    if isinstance(label, dict):
        if "labels" in label:
            label = label["labels"]
        elif len(label) == 1:
            label = next(iter(label.values()))
    label = np.asarray(label)
    if label.ndim != 2:
        label = np.squeeze(label)
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label array, got shape {label.shape} from {path}")
    return label


def remap_label(label: np.ndarray) -> np.ndarray:
    new_label = np.full(label.shape, IGNORE_INDEX, dtype=np.uint8)
    for old_id, new_id in MAPPING.items():
        new_label[label == old_id] = new_id
    return new_label


def clear_split_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def iter_stems(split_payload: dict) -> dict[str, list[str]]:
    splits = split_payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split.json is missing the 'splits' object")
    for split in ("train", "val", "test"):
        if split not in splits:
            raise ValueError(f"split.json is missing split '{split}'")
        if not isinstance(splits[split], list):
            raise ValueError(f"split '{split}' is not a list")
    return splits


def write_split_labels(src_label_dir: Path, dst_label_dir: Path, splits: dict[str, list[str]]) -> None:
    clear_split_dirs(dst_label_dir)

    total = 0
    for split_name in ("train", "val", "test"):
        stems = splits[split_name]
        split_out = dst_label_dir / split_name
        for stem in tqdm(stems, desc=f"remap {split_name}"):
            src_path = src_label_dir / f"{stem}_label.npy"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source label: {src_path}")
            dst_path = split_out / src_path.name

            label = load_label(src_path)
            new_label = remap_label(label)
            np.save(dst_path, new_label)
            total += 1

    print(f"Wrote {total} remapped labels to {dst_label_dir}")


def distribution(folder: Path) -> None:
    counter: Dict[int, int] = {}
    files = sorted(folder.glob("*.npy"))
    for path in tqdm(files, desc=f"stat {folder.name}"):
        label = np.load(path)
        unique, counts = np.unique(label, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist()):
            counter[value] = counter.get(value, 0) + count

    total = sum(counter.values())
    print(f"\n===== Distribution: {folder} =====")
    for key in sorted(counter):
        ratio = counter[key] / total if total else 0.0
        print(f"{key:3d}: {counter[key]:12d} ({ratio:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-label-dir",
        type=Path,
        default=Path("/mnt/data/mhy/dataset/Sen4Agri_31TCG/label"),
        help="Flat source label directory containing *_label.npy files",
    )
    parser.add_argument(
        "--dst-label-dir",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/label"),
        help="Destination label root with train/val/test subdirectories",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("/mnt/data/mhy/RSFM/AEF-seg-learning/data/Sen4Agri_31TCG/split.json"),
        help="Manifest describing train/val/test stems",
    )
    args = parser.parse_args()

    payload = json.loads(args.split_json.read_text(encoding="utf-8"))
    splits = iter_stems(payload)

    write_split_labels(args.src_label_dir, args.dst_label_dir, splits)

    for split_name in ("train", "val", "test"):
        distribution(args.dst_label_dir / split_name)


if __name__ == "__main__":
    main()