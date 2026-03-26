import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.data_gee_multisource import create_gee_multisource_dataloader


def parse_source_list(value: str) -> list[str]:
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("At least one input source is required")

    allowed = {"landsat", "sentinel1", "sentinel2"}
    invalid = [item for item in sources if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported input sources: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on GEE multi-source dataset and export 64D embeddings",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing GEE sample_*.npz files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt) from outputs_gee_multisource",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_gee_multisource/embeddings",
        help="Directory to save per-tile embedding npz files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (1 recommended to keep tile ordering)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch size used during training (H, W)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help=(
            "Model size of AlphaEarth used during training. "
            "Must match the model_size used in run_train_gee_multisource "
            "(e.g. 'tiny' for low-memory CPU runs)."
        ),
    )
    parser.add_argument(
        "--input_sources",
        type=str,
        default="auto",
        help=(
            "Input sources to use during inference. "
            "Use 'auto' to use all sources available in current dataset/checkpoint, "
            "or provide a comma-separated subset like 'landsat,sentinel2'."
        ),
    )
    parser.add_argument(
        "--summary_strategy",
        type=str,
        default="full_period",
        choices=["full_period", "per_timestamp"],
        help=(
            "How to summarize over time: "
            "'full_period' (single embedding per tile over whole valid_period) or "
            "'per_timestamp' (one embedding per time step for later time-series annotation alignment)"
        ),
    )
    parser.add_argument(
        "--max_time_steps",
        type=int,
        default=None,
        help=(
            "Optional cap on number of timestamps per sample when using "
            "'per_timestamp' summary_strategy. If set, only the first N time steps are used."
        ),
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    channel_map: Dict[str, int],
    device: torch.device,
    model_size: str,
) -> AlphaEarthFoundations:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    input_sources = model_config.get(
        "input_sources",
        channel_map,
    )
    decode_sources = model_config.get(
        "decode_sources",
        channel_map,
    )

    model = AlphaEarthFoundations(
        model_size=model_size,
        input_sources=input_sources,
        decode_sources=decode_sources,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def upsample_embedding_grid(emb: np.ndarray, target_hw: int) -> np.ndarray:
    """Upsample (H, W, C) embedding grid to (target_hw, target_hw, C)."""
    h, w, _ = emb.shape
    if h == target_hw and w == target_hw:
        return emb

    emb_t = torch.from_numpy(emb).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    emb_t = F.interpolate(
        emb_t,
        size=(target_hw, target_hw),
        mode="bilinear",
        align_corners=False,
    )
    return emb_t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def get_tile_id(tile_path: Path, fallback_idx: int) -> str:
    """Extract sample id from file name like sample_10000.npz; fallback to index."""
    stem = tile_path.stem  # e.g. sample_10000
    if stem.startswith("sample_"):
        tile_id = stem[len("sample_"):]
        if tile_id:
            return tile_id
    return f"{fallback_idx:04d}"


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataloader = create_gee_multisource_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=False,
    )

    # Infer available sources/channels from first sample
    first_sample = dataloader.dataset[0]
    available_sources = list(first_sample["source_data"].keys())
    channel_map = {
        source: first_sample["source_data"][source].shape[-1]
        for source in available_sources
    }

    model = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        channel_map=channel_map,
        device=device,
        model_size=args.model_size,
    )

    model_sources = list(model.input_sources.keys())
    if args.input_sources.strip().lower() == "auto":
        selected_sources = [src for src in model_sources if src in available_sources]
    else:
        requested_sources = parse_source_list(args.input_sources)
        selected_sources = [src for src in requested_sources if src in model_sources and src in available_sources]

    if not selected_sources:
        raise ValueError(
            "No usable input source remains after filtering by --input_sources, "
            "checkpoint model sources, and dataset availability."
        )

    print(f"Available dataset sources: {available_sources}")
    print(f"Model input sources: {model_sources}")
    print(f"Selected active sources: {selected_sources}")

    def build_model_inputs(batch: Dict[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raw_source_data: Dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in batch["source_data"].items()
        }
        raw_timestamps: Dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in batch["timestamps"].items()
        }

        template_src = next(iter(raw_source_data.keys()))
        template_x = raw_source_data[template_src]
        template_ts = raw_timestamps[template_src]

        source_data: Dict[str, torch.Tensor] = {}
        timestamps: Dict[str, torch.Tensor] = {}

        for src, channels in model.input_sources.items():
            if src in raw_source_data:
                x = raw_source_data[src]
                ts = raw_timestamps[src]
            else:
                B, T, H, W, _ = template_x.shape
                x = torch.zeros((B, T, H, W, channels), device=device, dtype=template_x.dtype)
                ts = template_ts

            if src not in selected_sources:
                x = torch.zeros_like(x)

            source_data[src] = x
            timestamps[src] = ts

        return source_data, timestamps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    num_samples = len(dataset)

    print(f"Running inference on {num_samples} samples from {data_dir}...")

    if args.summary_strategy == "full_period":
        # Original behaviour: one embedding per tile over its full valid_period
        global_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            source_data, timestamps = build_model_inputs(batch)
            valid_periods = batch["valid_periods"]

            out: Dict[str, Any] = model(
                source_data=source_data,
                timestamps=timestamps,
                valid_periods=valid_periods,
            )

            embeddings = out["embeddings"].detach().cpu().numpy()  # (B, H', W', 64)
            image_embeddings = out["image_embeddings"].detach().cpu().numpy()  # (B, 64)

            batch_size = embeddings.shape[0]
            for i in range(batch_size):
                sample_idx = global_idx
                global_idx += 1

                # Map back to file name in dataset
                tile_path = dataset.files[sample_idx]
                tile_id = get_tile_id(tile_path, sample_idx)

                emb = embeddings[i]  # (H', W', 64)
                emb_upsampled = upsample_embedding_grid(emb, args.patch_size)  # (patch, patch, 64)
                img_emb = image_embeddings[i]  # (64,)

                # Save timestamps from the first active source as reference time axis.
                reference_source = selected_sources[0]
                ts = batch["timestamps"][reference_source][i].detach().cpu().numpy()

                out_path = output_dir / f"embedding_{tile_id}.npz"
                np.savez(
                    out_path,
                    embeddings=emb_upsampled,
                    embeddings_native=emb,
                    image_embedding=img_emb,
                    timestamps=ts,
                    selected_sources=np.array(selected_sources, dtype=object),
                    tile_file=str(tile_path),
                )

                print(
                    f"Saved embeddings to {out_path} "
                    f"(tile: {tile_path.name}, emb shape {emb_upsampled.shape}, native {emb.shape})"
                )
    else:
        # per_timestamp: for each tile, compute an embedding for each time step.
        # This is useful when you later have time-series annotations and
        # want to align each annotation timestamp to a corresponding 64D embedding.
        if args.batch_size != 1:
            raise ValueError(
                "summary_strategy='per_timestamp' currently requires batch_size=1 "
                "so that tiles can be processed independently."
            )

        global_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            source_data, timestamps = build_model_inputs(batch)

            # Use the first active source as reference time axis.
            reference_source = selected_sources[0]
            ts_ref = timestamps[reference_source][0]  # (T,)
            T = ts_ref.shape[0]
            if args.max_time_steps is not None:
                T_eff = min(T, args.max_time_steps)
            else:
                T_eff = T

            per_ts_embeddings = []
            per_ts_img_embeddings = []

            for t_idx in range(T_eff):
                t_val = float(ts_ref[t_idx].item())
                valid_periods = [(t_val, t_val)]  # single-sample batch

                out: Dict[str, Any] = model(
                    source_data=source_data,
                    timestamps=timestamps,
                    valid_periods=valid_periods,
                )

                emb_t = out["embeddings"][0].detach().cpu().numpy()  # (H', W', 64)
                img_emb_t = (
                    out["image_embeddings"][0].detach().cpu().numpy()
                )  # (64,)

                per_ts_embeddings.append(emb_t)
                per_ts_img_embeddings.append(img_emb_t)

            per_ts_embeddings_arr = np.stack(per_ts_embeddings, axis=0)
            per_ts_embeddings_upsampled = np.stack(
                [upsample_embedding_grid(emb_t, args.patch_size) for emb_t in per_ts_embeddings],
                axis=0,
            )
            per_ts_img_arr = np.stack(per_ts_img_embeddings, axis=0)
            ts_arr = ts_ref[:T_eff].detach().cpu().numpy()

            sample_idx = global_idx
            global_idx += 1
            tile_path = dataset.files[sample_idx]
            tile_id = get_tile_id(tile_path, sample_idx)

            out_path = output_dir / f"embedding_timeseries_{tile_id}.npz"
            np.savez(
                out_path,
                embeddings_per_time=per_ts_embeddings_upsampled,
                embeddings_per_time_native=per_ts_embeddings_arr,
                image_embeddings_per_time=per_ts_img_arr,
                timestamps=ts_arr,
                selected_sources=np.array(selected_sources, dtype=object),
                tile_file=str(tile_path),
            )

            print(
                f"Saved time-series embeddings to {out_path} "
                f"(tile: {tile_path.name}, T={T_eff}, "
                f"emb shape {per_ts_embeddings_upsampled.shape[1:]}, "
                f"native {per_ts_embeddings_arr.shape[1:]})"
            )

    print("Inference finished.")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
