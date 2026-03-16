import argparse
from pathlib import Path
from typing import Dict

import torch

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.training import create_trainer
from alphaearth.data_gee_multisource import create_gee_multisource_dataloader


def parse_reconstruction_sources(value: str) -> list[str]:
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("At least one reconstruction source is required")

    allowed = {"landsat", "sentinel1", "sentinel2"}
    invalid = [item for item in sources if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported reconstruction sources: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaEarth on GEE L8/S1/S2 multi-source dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/gee_multisource",
        help="Directory containing GEE .npz samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Spatial patch size (H, W)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --epochs if set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for learning rate",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_gee_multisource",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss term (default 1.0)",
    )
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=0.01,
        help="Weight for uniformity loss term (default 0.01)",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.005,
        help="Weight for teacher-student consistency loss term (default 0.005)",
    )
    parser.add_argument(
        "--detail_weight",
        type=float,
        default=0.05,
        help="Weight for gradient/detail reconstruction loss term (default 0.05).",
    )
    parser.add_argument(
        "--uniformity_ramp_steps",
        type=int,
        default=0,
        help="Linearly ramp uniformity weight from 0 to target over this many steps (0 disables ramp).",
    )
    parser.add_argument(
        "--consistency_ramp_steps",
        type=int,
        default=0,
        help="Linearly ramp consistency weight from 0 to target over this many steps (0 disables ramp).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help=(
            "Model size for AlphaEarth encoder/decoder. "
            "Use 'tiny' on CPU or low-memory machines to reduce parameter "
            "count and memory usage. Default is 'small'."
        ),
    )
    parser.add_argument(
        "--reconstruction_sources",
        type=parse_reconstruction_sources,
        default=["landsat", "sentinel1", "sentinel2"],
        help=(
            "Comma-separated list of sources to reconstruct. "
            "By default all three sources are reconstructed, while Landsat/S1/S2 are always concatenated "
            "as encoder inputs before STP processing."
        ),
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable mixed precision training on CUDA (1=on, 0=off).",
    )
    parser.add_argument(
        "--grad_checkpoint",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable gradient checkpointing in STP encoder blocks (1=on, 0=off).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--resume_latest",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, resume from <output_dir>/checkpoint_latest.pt when it exists.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading GEE multi-source dataset from {data_dir}")

    dataloader = create_gee_multisource_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=True,
    )

    dataset_size = len(dataloader.dataset)
    steps_per_epoch = dataset_size // args.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Dataset too small for given batch size.")

    if args.max_steps is None:
        max_steps = args.epochs * steps_per_epoch
    else:
        max_steps = args.max_steps

    print(f"Dataset: {dataset_size} samples, {steps_per_epoch} steps/epoch")
    print(f"Training for {max_steps} steps ({max_steps / steps_per_epoch:.2f} epochs)")

    # 自动根据数据集确定可用的数据源与通道数，兼容仅 S1/S2 的数据
    first_sample = dataloader.dataset[0]
    available_sources = list(first_sample["source_data"].keys())
    channel_map = {name: first_sample["source_data"][name].shape[-1] for name in available_sources}

    print(f"Available sources in dataset: {available_sources}")

    decode_sources: Dict[str, int] = {}
    for name in args.reconstruction_sources:
        if name in channel_map:
            decode_sources[name] = channel_map[name]
        else:
            print(f"Warning: requested reconstruction source '{name}' not present in dataset; ignoring.")

    if not decode_sources:
        raise ValueError(
            "None of the requested reconstruction_sources are present in the dataset. "
            "Please check --reconstruction_sources and the GEE samples."
        )

    model = AlphaEarthFoundations(
        model_size=args.model_size,
        input_sources=channel_map,
        decode_sources=decode_sources,
    )
    model.encoder.use_gradient_checkpointing = bool(args.grad_checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Model size: {args.model_size}")
    print(f"Input sources: {model.input_sources}")
    print(f"Decode sources: {model.decode_sources}")
    print(f"AMP enabled: {bool(args.amp)}")
    print(f"Gradient checkpointing enabled: {bool(args.grad_checkpoint)}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    param_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size (float32): {param_size_mb:.2f} MB")
    print("\n" + "-" * 80)
    print("MODEL ARCHITECTURE")
    print("-" * 80)
    print(model)
    print("=" * 80 + "\n")

    trainer = create_trainer(
        model=model,
        dataloader=dataloader,
        text_adapter=None,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        reconstruction_weight=args.reconstruction_weight,
        uniformity_weight=args.uniformity_weight,
        consistency_weight=args.consistency_weight,
        detail_weight=args.detail_weight,
        use_amp=bool(args.amp),
        uniformity_ramp_steps=args.uniformity_ramp_steps,
        consistency_ramp_steps=args.consistency_ramp_steps,
    )

    trainer.max_steps = max_steps
    trainer.warmup_steps = args.warmup_steps

    resume_path = None
    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
    elif bool(args.resume_latest):
        candidate = Path(args.output_dir) / 'checkpoint_latest.pt'
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        resumed_step = trainer.load_checkpoint(str(resume_path), load_optimizer=True)
        print(f"Resumed from checkpoint: {resume_path} (step={resumed_step})")

    print(f"Starting training for {max_steps} steps...")
    print(f"Output directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(max_steps=max_steps, log_every=args.log_every)

    print("Training run finished.")


if __name__ == "__main__":
    main()
