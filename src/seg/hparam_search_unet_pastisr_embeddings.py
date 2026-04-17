import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Callable, List, Sequence

from seg import train_unet_pastisr_embeddings as base_train


def _parse_list(raw: str, cast: Callable[[str], object]) -> List[object]:
    out: List[object] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out


def _parse_search_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    search_parser = argparse.ArgumentParser(add_help=False)
    search_parser.add_argument("--search_lrs", type=str, default="1e-4,3e-4")
    search_parser.add_argument("--search_dice_weights", type=str, default="")
    search_parser.add_argument("--search_crop_sizes", type=str, default="")
    search_parser.add_argument("--search_model_variants", type=str, default="")
    search_parser.add_argument("--search_unet_depths", type=str, default="")
    search_parser.add_argument("--search_base_channels", type=str, default="")
    search_parser.add_argument("--search_use_fg_sampler", type=str, default="")
    search_parser.add_argument("--search_ignore_bg_dice", type=str, default="")
    search_parser.add_argument("--search_ignore_bg_metrics", type=str, default="")
    search_parser.add_argument("--search_focal_focus_classes", type=str, default="")
    search_parser.add_argument("--search_tag", type=str, default="search")
    search_parser.add_argument(
        "--objective",
        type=str,
        default="best_val_miou",
        choices=["best_val_miou", "best_val_acc", "best_val_loss"],
        help="Metric used to select best hyper-parameter set.",
    )
    search_parser.add_argument(
        "--max_trials",
        type=int,
        default=0,
        help="If >0, stop after max_trials combinations (for quick dry-runs).",
    )

    search_args, remaining = search_parser.parse_known_args()
    base_args = base_train.parse_args(remaining)
    return search_args, base_args


def _one_or_search(raw: str, base_value: object, cast: Callable[[str], object]) -> List[object]:
    values = _parse_list(raw, cast)
    return values if values else [base_value]


def main() -> None:
    search_args, base_args = _parse_search_args()

    lr_list = _one_or_search(search_args.search_lrs, float(base_args.lr), float)
    dice_list = _one_or_search(search_args.search_dice_weights, float(base_args.dice_weight), float)
    crop_list = _one_or_search(search_args.search_crop_sizes, int(base_args.train_crop_size), int)
    model_variant_list = _one_or_search(search_args.search_model_variants, str(base_args.model_variant), str)
    unet_depth_list = _one_or_search(search_args.search_unet_depths, int(base_args.unet_depth), int)
    base_channels_list = _one_or_search(search_args.search_base_channels, int(base_args.base_channels), int)
    use_fg_sampler_list = _one_or_search(search_args.search_use_fg_sampler, int(base_args.use_fg_sampler), int)
    ignore_bg_dice_list = _one_or_search(
        search_args.search_ignore_bg_dice,
        int(base_args.ignore_background_in_dice),
        int,
    )
    ignore_bg_metrics_list = _one_or_search(
        search_args.search_ignore_bg_metrics,
        int(base_args.ignore_background_in_metrics),
        int,
    )
    focal_focus_classes_list = _one_or_search(
        search_args.search_focal_focus_classes,
        str(base_args.focal_focus_classes),
        str,
    )

    all_combos: Sequence[tuple[object, ...]] = list(
        itertools.product(
            lr_list,
            dice_list,
            crop_list,
            model_variant_list,
            unet_depth_list,
            base_channels_list,
            use_fg_sampler_list,
            ignore_bg_dice_list,
            ignore_bg_metrics_list,
            focal_focus_classes_list,
        )
    )

    if not all_combos:
        raise RuntimeError("No hyper-parameter combinations generated.")

    root_out = Path(base_args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    print("==== Pastis-R Embedding Hyper-Parameter Search ====")
    print(f"Total combinations: {len(all_combos)}")
    print(f"Objective: {search_args.objective}")
    if search_args.max_trials > 0:
        print(f"Max trials: {search_args.max_trials}")

    results = []
    best_cfg = None
    if search_args.objective == "best_val_loss":
        best_score = float("inf")
    else:
        best_score = float("-inf")

    for idx, combo in enumerate(all_combos, start=1):
        if search_args.max_trials > 0 and idx > search_args.max_trials:
            print(f"Reached max_trials={search_args.max_trials}, stop.")
            break

        (
            lr,
            dice_w,
            crop_sz,
            model_variant,
            unet_depth,
            base_channels,
            use_fg_sampler,
            ignore_bg_dice,
            ignore_bg_metrics,
            focal_focus_classes,
        ) = combo

        trial_args = copy.deepcopy(base_args)
        trial_args.lr = float(lr)
        trial_args.dice_weight = float(dice_w)
        trial_args.train_crop_size = int(crop_sz)
        trial_args.model_variant = str(model_variant)
        trial_args.unet_depth = int(unet_depth)
        trial_args.base_channels = int(base_channels)
        trial_args.use_fg_sampler = int(use_fg_sampler)
        trial_args.ignore_background_in_dice = int(ignore_bg_dice)
        trial_args.ignore_background_in_metrics = int(ignore_bg_metrics)
        trial_args.focal_focus_classes = str(focal_focus_classes)

        subdir_name = (
            f"{search_args.search_tag}_"
            f"lr{float(lr):g}_dice{float(dice_w):g}_crop{int(crop_sz)}_"
            f"mv{model_variant}_d{int(unet_depth)}_ch{int(base_channels)}_"
            f"fgs{int(use_fg_sampler)}_igd{int(ignore_bg_dice)}_igm{int(ignore_bg_metrics)}"
        )
        trial_args.output_dir = str(root_out / subdir_name)

        print("\n---- Trial {}/{} ----".format(idx, len(all_combos)))
        print(
            "cfg:",
            {
                "lr": lr,
                "dice_weight": dice_w,
                "train_crop_size": crop_sz,
                "model_variant": model_variant,
                "unet_depth": unet_depth,
                "base_channels": base_channels,
                "use_fg_sampler": use_fg_sampler,
                "ignore_background_in_dice": ignore_bg_dice,
                "ignore_background_in_metrics": ignore_bg_metrics,
                "focal_focus_classes": focal_focus_classes,
            },
        )
        print(f"output_dir: {trial_args.output_dir}")

        metrics = base_train.train(trial_args)

        trial_result = {
            "trial_index": idx,
            "output_dir": trial_args.output_dir,
            "lr": float(lr),
            "dice_weight": float(dice_w),
            "train_crop_size": int(crop_sz),
            "model_variant": str(model_variant),
            "unet_depth": int(unet_depth),
            "base_channels": int(base_channels),
            "use_fg_sampler": int(use_fg_sampler),
            "ignore_background_in_dice": int(ignore_bg_dice),
            "ignore_background_in_metrics": int(ignore_bg_metrics),
            "focal_focus_classes": str(focal_focus_classes),
        }
        if isinstance(metrics, dict):
            trial_result.update(metrics)

        score = float(trial_result.get(search_args.objective, float("nan")))
        trial_result["objective"] = search_args.objective
        trial_result["objective_score"] = score
        results.append(trial_result)

        print(f"objective({search_args.objective}) = {score:.6f}")

        if search_args.objective == "best_val_loss":
            is_better = score < best_score
        else:
            is_better = score > best_score

        if is_better:
            best_score = score
            best_cfg = trial_result
            print("New best config found.")

    summary = {
        "search_tag": search_args.search_tag,
        "objective": search_args.objective,
        "num_trials": len(results),
        "best": best_cfg,
        "results": results,
    }
    summary_path = root_out / f"hparam_search_{search_args.search_tag}_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Search Finished ====")
    print(f"Summary saved to: {summary_path}")
    if best_cfg is not None:
        print("Best config:")
        print(json.dumps(best_cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
