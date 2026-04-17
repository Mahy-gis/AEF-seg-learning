import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import List, Optional

from seg import train_unet_from_embeddings as base_train


def _parse_search_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """解析搜索相关参数 + 原始训练参数。

    用法示例：
      python -m seg.hparam_search_unet_from_embeddings \
        --search_lrs 1e-4,3e-4,5e-4 \
        --search_dice_weights 0.2,0.4,0.6 \
        --search_crop_sizes 96,128 \
        --embeddings_path ... --labels_file ... --output_dir ... 其它原 train 参数...
    """
    search_parser = argparse.ArgumentParser(add_help=False)
    search_parser.add_argument(
        "--search_lrs",
        type=str,
        default="5e-5,1e-4,3e-4,5e-4,1e-3",
        help="逗号分隔的学习率候选列表，用于网格搜索 (默认: 5e-5,1e-4,3e-4,5e-4,1e-3)",
    )
    search_parser.add_argument(
        "--search_dice_weights",
        type=str,
        default="",
        help="可选：逗号分隔的 dice_weight 候选列表；为空则仅使用原始 --dice_weight",
    )
    search_parser.add_argument(
        "--search_crop_sizes",
        type=str,
        default="",
        help="可选：逗号分隔的 train_crop_size 候选列表；为空则仅使用原始 --train_crop_size",
    )
    search_parser.add_argument(
        "--search_tag",
        type=str,
        default="search",
        help="搜索实验的标签名，会出现在子目录与结果文件名中 (默认 'search')",
    )

    # 先吃掉搜索相关参数，其余参数留给原始 train 脚本的 parse_args 解析
    search_args, remaining = search_parser.parse_known_args()
    base_args = base_train.parse_args(remaining)
    return search_args, base_args


def _parse_list(s: str, cast_func):
    items: List = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        items.append(cast_func(tok))
    return items


def main() -> None:
    search_args, base_args = _parse_search_args()

    # 搜索空间
    lr_list = _parse_list(search_args.search_lrs, float)
    if not lr_list:
        raise ValueError("--search_lrs 至少需要一个有效的学习率")

    if search_args.search_dice_weights:
        dice_list = _parse_list(search_args.search_dice_weights, float)
    else:
        dice_list = [float(base_args.dice_weight)]

    if search_args.search_crop_sizes:
        crop_list = _parse_list(search_args.search_crop_sizes, int)
    else:
        crop_list = [int(base_args.train_crop_size)]

    root_out = Path(base_args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    results = []
    best_cfg = None
    best_miou = -1.0

    print("==== 超参搜索开始 ====")
    print(f"学习率 candidates: {lr_list}")
    print(f"dice_weight candidates: {dice_list}")
    print(f"train_crop_size candidates: {crop_list}")
    print(f"统一输出根目录: {root_out}")

    for lr, dice_w, crop_sz in itertools.product(lr_list, dice_list, crop_list):
        trial_args = copy.deepcopy(base_args)
        trial_args.lr = float(lr)
        trial_args.dice_weight = float(dice_w)
        trial_args.train_crop_size = int(crop_sz)

        subdir_name = f"{search_args.search_tag}_lr{lr:g}_dice{dice_w:g}_crop{crop_sz}"
        trial_args.output_dir = str(root_out / subdir_name)

        print("\n==== 运行配置 ====")
        print(f"  lr={lr:g}, dice_weight={dice_w:g}, train_crop_size={crop_sz}")
        print(f"  output_dir={trial_args.output_dir}")

        metrics = base_train.train(trial_args)
        # metrics 由 train() 返回，包括 best_val_miou / best_val_loss / best_val_acc
        trial_result = {
            "lr": float(lr),
            "dice_weight": float(dice_w),
            "train_crop_size": int(crop_sz),
            "output_dir": trial_args.output_dir,
        }
        if isinstance(metrics, dict):
            trial_result.update(metrics)

        results.append(trial_result)

        cur_miou = float(trial_result.get("best_val_miou", -1.0))
        if cur_miou > best_miou:
            best_miou = cur_miou
            best_cfg = trial_result

        print("  -> 本次 best_val_mIoU = {:.4f}".format(cur_miou))

    # 保存搜索结果
    summary_path = root_out / f"hparam_search_{search_args.search_tag}_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best_cfg}, f, ensure_ascii=False, indent=2)

    print("\n==== 搜索完成 ====")
    print(f"结果已保存到: {summary_path}")
    if best_cfg is not None: 
        print("最优配置:")
        print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    else:
        print("未得到任何有效结果，请检查验证集设置是否正确。")


if __name__ == "__main__":
    main()
