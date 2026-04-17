import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_COLUMNS = [
    "run_dir",
    "script",
    "timestamp",
    "best_val_miou",
    "best_val_acc",
    "best_val_loss",
    "last_val_miou",
    "last_train_miou",
    "duration_sec",
    "epochs_ran",
    "model_variant",
    "unet_depth",
    "base_channels",
    "lr",
    "batch_size",
    "dice_weight",
    "dataset_format",
    "time_steps",
    "use_fg_sampler",
    "ignore_index",
    "background_index",
]


def _get_nested(d: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_float_for_sort(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("-inf")


def _format_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def _discover_summary_files(roots: Sequence[Path], pattern: str) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.is_file() and root.name == pattern:
            out.append(root)
            continue
        if root.is_dir():
            out.extend(sorted(root.rglob(pattern)))
    return out


def _build_row_from_eval_summary(eval_summary: Dict[str, Any], summary_path: Path, root_for_rel: Path) -> Dict[str, Any]:
    try:
        run_dir = str(summary_path.parent.relative_to(root_for_rel))
    except ValueError:
        run_dir = str(summary_path.parent)

    return {
        "run_dir": run_dir,
        "script": "eval_unet_from_embeddings.py",
        "timestamp": "",
        "best_val_miou": eval_summary.get("val_miou", ""),
        "best_val_acc": eval_summary.get("val_acc", ""),
        "best_val_loss": eval_summary.get("val_loss", ""),
        "last_val_miou": eval_summary.get("val_miou", ""),
        "last_train_miou": "",
        "duration_sec": "",
        "epochs_ran": "",
        "epochs_target": "",
        "model_variant": eval_summary.get("model_variant", ""),
        "unet_depth": eval_summary.get("unet_depth", ""),
        "base_channels": eval_summary.get("base_channels", ""),
        "lr": "",
        "batch_size": "",
        "dice_weight": "",
        "dataset_format": "",
        "time_steps": "",
        "use_fg_sampler": "",
        "ignore_index": "",
        "background_index": "",
        "summary_path": str(summary_path),
    }


def _build_row(summary: Dict[str, Any], summary_path: Path, root_for_rel: Path) -> Dict[str, Any]:
    best = _get_nested(summary, "best_metrics", {})
    last = _get_nested(summary, "last_metrics", {})
    model = _get_nested(summary, "model", {})
    optim = _get_nested(summary, "optimization", {})
    labels = _get_nested(summary, "labels", {})
    data = _get_nested(summary, "data", {})

    try:
        run_dir = str(summary_path.parent.relative_to(root_for_rel))
    except ValueError:
        run_dir = str(summary_path.parent)

    return {
        "run_dir": run_dir,
        "script": summary.get("script", ""),
        "timestamp": summary.get("timestamp", ""),
        "best_val_miou": best.get("best_val_miou", ""),
        "best_val_acc": best.get("best_val_acc", ""),
        "best_val_loss": best.get("best_val_loss", ""),
        "last_val_miou": last.get("val_miou", ""),
        "last_train_miou": last.get("train_miou", ""),
        "duration_sec": summary.get("duration_sec", ""),
        "epochs_ran": summary.get("epochs_ran", ""),
        "epochs_target": summary.get("epochs_target", ""),
        "model_variant": model.get("model_variant", ""),
        "unet_depth": model.get("unet_depth", ""),
        "base_channels": model.get("base_channels", ""),
        "lr": optim.get("lr", _get_nested(summary, "args.lr", "")),
        "batch_size": optim.get("batch_size", _get_nested(summary, "args.batch_size", "")),
        "dice_weight": optim.get("dice_weight", _get_nested(summary, "args.dice_weight", "")),
        "dataset_format": data.get("dataset_format", _get_nested(summary, "args.dataset_format", "")),
        "time_steps": data.get("time_steps", _get_nested(summary, "args.time_steps", "")),
        "use_fg_sampler": data.get("use_fg_sampler", _get_nested(summary, "args.use_fg_sampler", "")),
        "ignore_index": labels.get("ignore_index", _get_nested(summary, "args.ignore_index", "")),
        "background_index": labels.get("background_index", _get_nested(summary, "args.background_index", "")),
        "summary_path": str(summary_path),
    }


def _write_csv(rows: Sequence[Dict[str, Any]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_cell(row.get(k, "")) for k in columns})


def _write_markdown(rows: Sequence[Dict[str, Any]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = [_format_cell(row.get(k, "")).replace("|", "\\|") for k in columns]
            f.write("| " + " | ".join(vals) + " |\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run_summary.json into a comparison table.")
    p.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more root dirs to recursively scan for run_summary.json.",
    )
    p.add_argument("--pattern", type=str, default="run_summary.json")
    p.add_argument(
        "--fallback_eval_summary",
        type=int,
        default=1,
        choices=[0, 1],
        help="If no --pattern files are found, fallback to scanning eval_summary.json (default 1).",
    )
    p.add_argument("--sort_by", type=str, default="best_val_miou")
    p.add_argument("--ascending", type=int, default=0, choices=[0, 1])
    p.add_argument("--top_k", type=int, default=0, help="Keep top K rows after sorting (0 keeps all).")
    p.add_argument(
        "--columns",
        type=str,
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated output columns.",
    )
    p.add_argument(
        "--extra_json_fields",
        type=str,
        default="",
        help="Comma-separated nested JSON paths (e.g., args.focal_focus_classes,args.norm_type).",
    )
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    roots = [Path(x).expanduser().resolve() for x in args.roots]
    summary_files = _discover_summary_files(roots, args.pattern)
    source_mode = "run_summary"
    if not summary_files and args.fallback_eval_summary == 1:
        summary_files = _discover_summary_files(roots, "eval_summary.json")
        source_mode = "eval_summary"
    if not summary_files:
        raise FileNotFoundError(
            "No summary files found. Check --roots/--pattern. "
            "Tip: rerun training with updated scripts to generate run_summary.json, "
            "or enable --fallback_eval_summary 1."
        )

    rows: List[Dict[str, Any]] = []
    for sp in summary_files:
        with sp.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        best_root = sp.parent
        for root in roots:
            if str(sp).startswith(str(root)):
                best_root = root
                break

        if source_mode == "run_summary":
            row = _build_row(summary, sp, best_root)
        else:
            row = _build_row_from_eval_summary(summary, sp, best_root)

        extra_fields = [x.strip() for x in str(args.extra_json_fields).split(",") if x.strip()]
        for ef in extra_fields:
            row[ef] = _get_nested(summary, ef, "")

        rows.append(row)

    rows.sort(key=lambda r: _to_float_for_sort(r.get(args.sort_by, "")), reverse=(args.ascending == 0))

    if args.top_k > 0:
        rows = rows[: args.top_k]

    columns = [x.strip() for x in args.columns.split(",") if x.strip()]
    extra_fields = [x.strip() for x in str(args.extra_json_fields).split(",") if x.strip()]
    for ef in extra_fields:
        if ef not in columns:
            columns.append(ef)

    first_root = roots[0]
    output_csv = Path(args.output_csv) if args.output_csv else first_root / "experiment_compare.csv"
    output_md = Path(args.output_md) if args.output_md else first_root / "experiment_compare.md"

    _write_csv(rows, columns, output_csv)
    _write_markdown(rows, columns, output_md)

    print(f"Summary source mode: {source_mode}")
    print(f"Found summaries: {len(summary_files)}")
    print(f"Rows in table: {len(rows)}")
    print(f"CSV saved to: {output_csv}")
    print(f"Markdown saved to: {output_md}")


if __name__ == "__main__":
    main()
