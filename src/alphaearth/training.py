from typing import Any, Dict, List, Optional, Tuple
import itertools
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.loss_function import AEFLoss



class Trainer:
    def __init__(self,
                 model: AlphaEarthFoundations,
                 dataloader,
                 text_adapter = None, 
                 lr: float = 1e-4,
                 device: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 reconstruction_weight: float = 1.0,
                 uniformity_weight: float = 0.01,
                 consistency_weight: float = 0.005,
                 detail_weight: float = 0.05,
                 text_weight: float = 0.001,
                 use_amp: bool = True,
                 uniformity_ramp_steps: int = 0,
                 consistency_ramp_steps: int = 0):
        self.model = model
        self.dataloader = dataloader
        self.text_adapter = text_adapter
        # Allow adjusting loss term weights to trade off reconstruction vs.
        # representation regularization, which is useful when focusing on
        # good reconstructions (e.g., for downstream annotation matching).
        self.loss_fn = AEFLoss(
            reconstruction_weight=reconstruction_weight,
            uniformity_weight=uniformity_weight,
            consistency_weight=consistency_weight,
            detail_weight=detail_weight,
            text_weight=text_weight,
        )
        self.base_uniformity_weight = uniformity_weight
        self.base_consistency_weight = consistency_weight
        self.uniformity_ramp_steps = max(0, int(uniformity_ramp_steps))
        self.consistency_ramp_steps = max(0, int(consistency_ramp_steps))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = bool(use_amp) and self.device.type == 'cuda'
        self.model.to(self.device)
        if self.text_adapter is not None:
            self.text_adapter.to(self.device)
        params = list(self.model.parameters())
        if self.text_adapter is not None and any(p.requires_grad for p in self.text_adapter.parameters()):
            params += [p for p in self.text_adapter.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.output_dir = output_dir
        self.max_steps = 1000
        self.warmup_steps = 0
        self.start_step = 0
        self._visualization_batches: Optional[Dict[str, Dict[str, Any]]] = None
        # Track losses for visualization
        self.loss_history = {
            'steps': [],
            'total': [],
            'reconstruction': [],
            'detail': [],
            'uniformity': [],
            'consistency': [],
            'clip': [],
        }

    def _prepare_reconstruction_targets(
        self,
        batch: Dict[str, Any],
        src_key: str | None = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Prepare reconstruction targets for a given source.

        src_key:
            If provided, use this source from batch['source_data']; otherwise
            fall back to the first available source (for backwards compatibility
            with single-source training).
        """
        if src_key is None:
            src_key = next(iter(batch['source_data'].keys()))

        x = batch['source_data'][src_key].to(self.device)
        ts = batch['timestamps'][src_key].to(self.device)
        frame_valid_mask = batch.get('frame_valid_mask', {}).get(src_key)
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=self.device)
        else:
            frame_valid_mask = frame_valid_mask.to(self.device).bool()

        B, T, H, W, C = x.shape
        valid_counts = frame_valid_mask.sum(dim=1, keepdim=True)
        safe_counts = valid_counts.clamp_min(1)
        center = (ts * frame_valid_mask.float()).sum(dim=1, keepdim=True) / safe_counts.float()
        distances = (ts - center).abs()
        masked_distances = distances.masked_fill(~frame_valid_mask, float('inf'))
        idx = masked_distances.argmin(dim=1)

        no_valid = valid_counts.squeeze(1) == 0
        if no_valid.any():
            fallback_idx = (ts - ts.mean(dim=1, keepdim=True)).abs().argmin(dim=1)
            idx = torch.where(no_valid, fallback_idx, idx)

        batch_indices = torch.arange(B, device=self.device)
        target = x[batch_indices, idx]
        if spatial_size is not None and tuple(target.shape[1:3]) != tuple(spatial_size):
            target_2d = rearrange(target, 'b h w c -> b c h w')
            target_2d = F.interpolate(target_2d, size=spatial_size, mode='bilinear', align_corners=False)
            target = rearrange(target_2d, 'b c h w -> b h w c')
        pixel_mask = (target.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        target_ts = ts[batch_indices, idx]
        return {src_key: target}, {src_key: pixel_mask}, {src_key: target_ts}

    def _select_reconstruction_data(
        self,
        batch: Dict[str, Any],
        predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        targets: Dict[str, torch.Tensor] = {}
        target_masks: Dict[str, torch.Tensor] = {}
        target_timestamps: Dict[str, torch.Tensor] = {}

        decode_sources = predictions.keys() if predictions is not None else self.model.decode_sources.keys()
        for src_name in decode_sources:
            spatial_size = None
            if predictions is not None and src_name in predictions:
                spatial_size = tuple(predictions[src_name].shape[1:3])

            target_dict, mask_dict, ts_dict = self._prepare_reconstruction_targets(
                batch,
                src_key=src_name,
                spatial_size=spatial_size,
            )
            if mask_dict[src_name].sum().item() <= 0:
                continue

            targets[src_name] = target_dict[src_name]
            target_masks[src_name] = mask_dict[src_name]
            target_timestamps[src_name] = ts_dict[src_name]

        return targets, target_masks, target_timestamps

    def _prepare_visualization_batches(self) -> Dict[str, Dict[str, Any]]:
        if self._visualization_batches is not None:
            return self._visualization_batches

        preview_batches: Dict[str, Dict[str, Any]] = {}
        dataset = self.dataloader.dataset
        collate_fn = getattr(self.dataloader, 'collate_fn', None)
        if collate_fn is None:
            self._visualization_batches = preview_batches
            return preview_batches

        decode_sources = list(self.model.decode_sources.keys())
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_frame_masks = sample.get('frame_valid_mask', {})
            valid_sources = [
                src for src in decode_sources
                if (
                    src in sample_frame_masks and bool(sample_frame_masks[src].any())
                ) or (
                    src not in sample_frame_masks and bool(sample['source_data'][src].abs().sum() > 0)
                )
            ]
            if not valid_sources:
                continue

            batch = collate_fn([sample])
            for src in valid_sources:
                preview_batches.setdefault(src, batch)

            if all(src in preview_batches for src in decode_sources):
                break

        self._visualization_batches = preview_batches
        return preview_batches

    def _run_reconstruction_preview(
        self,
        batch: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        source_data: Dict[str, torch.Tensor] = {
            k: v.to(self.device) for k, v in batch['source_data'].items()
        }
        timestamps: Dict[str, torch.Tensor] = {
            k: v.to(self.device) for k, v in batch['timestamps'].items()
        }
        frame_valid_masks: Dict[str, torch.Tensor] = {
            k: v.to(self.device) for k, v in batch.get('frame_valid_mask', {}).items()
        }
        targets, target_masks, target_timestamps = self._select_reconstruction_data(batch)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                source_data,
                timestamps,
                batch['valid_periods'],
                temporal_masks=frame_valid_masks,
                decode_timestamps=target_timestamps,
            )
            predictions = {src: rec[:, 0].detach().cpu() for src, rec in out['reconstructions'].items() if src in targets}

        if was_training:
            self.model.train()

        targets_cpu = {src: tensor.detach().cpu() for src, tensor in targets.items() if src in predictions}
        masks_cpu = {src: tensor.detach().cpu() for src, tensor in target_masks.items() if src in predictions}
        return predictions, targets_cpu, masks_cpu

    def _to_display_rgb(self, src_name: str, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            gray = arr
            return np.stack([gray, gray, gray], axis=-1)

        if src_name in {'landsat', 'sentinel2'} and arr.shape[-1] >= 3:
            return arr[..., [2, 1, 0]]
        if arr.shape[-1] >= 3:
            return arr[..., :3]

        gray = arr[..., 0]
        return np.stack([gray, gray, gray], axis=-1)

    def _stretch_for_display(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
        span = hi - lo

        if np.all(span < 1e-6):
            lo = rgb.min(axis=(0, 1), keepdims=True)
            hi = rgb.max(axis=(0, 1), keepdims=True)
            span = hi - lo

        if np.all(span < 1e-6):
            fill_value = 0.35 if np.abs(rgb).max() < 1e-6 else 0.5
            return np.full_like(rgb, fill_value, dtype=np.float32)

        stretched = np.clip((rgb - lo) / (span + 1e-6), 0.0, 1.0)
        return np.power(stretched, 0.85)

    def train(self, max_steps: Optional[int] = None, log_every: int = 20):
        steps = max_steps or self.max_steps
        if self.start_step >= steps:
            print(
                f"Resume step ({self.start_step}) is already >= max_steps ({steps}); "
                "nothing to train. Increase --max_steps to continue."
            )
            return

        self.model.train()
        data_iter = itertools.cycle(self.dataloader)

        pbar = tqdm(range(self.start_step + 1, steps + 1), desc="Training", unit="step")
        start_time = time.time()
        
        for step in pbar:
            # Gradually enable regularization so early training focuses on reconstruction.
            if self.uniformity_ramp_steps > 0:
                uni_scale = min(1.0, step / float(self.uniformity_ramp_steps))
                self.loss_fn.uniformity_weight = self.base_uniformity_weight * uni_scale
            else:
                self.loss_fn.uniformity_weight = self.base_uniformity_weight

            if self.consistency_ramp_steps > 0:
                cons_scale = min(1.0, step / float(self.consistency_ramp_steps))
                self.loss_fn.consistency_weight = self.base_consistency_weight * cons_scale
            else:
                self.loss_fn.consistency_weight = self.base_consistency_weight

            batch = next(data_iter)
            step_start_time = time.time()
            
            source_data: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['source_data'].items()
            }
            timestamps: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch['timestamps'].items()
            }
            frame_valid_masks: Dict[str, torch.Tensor] = {
                k: v.to(self.device) for k, v in batch.get('frame_valid_mask', {}).items()
            }
            valid_periods: List[Tuple[float, float]] = batch['valid_periods']

            
            preselected_targets, preselected_masks, target_timestamps = self._select_reconstruction_data(batch)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                out = self.model(
                    source_data,
                    timestamps,
                    valid_periods,
                    temporal_masks=frame_valid_masks,
                    decode_timestamps=target_timestamps,
                )

                predictions: Dict[str, torch.Tensor] = {}
                for src, rec in out['reconstructions'].items():
                    predictions[src] = rec[:, 0]

                targets = {src: target for src, target in preselected_targets.items() if src in predictions}
                target_masks = {src: mask for src, mask in preselected_masks.items() if src in predictions}
                predictions = {src: pred for src, pred in predictions.items() if src in targets}

                # Optional text embeddings for text-image alignment loss
                text_embeddings = None
                if self.text_adapter is not None and 'texts' in batch:
                    text_embeddings = self.text_adapter.encode(batch['texts'], device=self.device)

                outputs_for_loss: Dict[str, Any] = {
                    'embeddings': out['embeddings'],
                    'teacher_embeddings': out['teacher_embeddings'],
                    'student_embeddings': out['student_embeddings'],
                    'image_embeddings': out['image_embeddings'],
                    'predictions': predictions,
                    'targets': targets,
                    'masks': target_masks,
                }
                if text_embeddings is not None and text_embeddings.shape[0] == out['image_embeddings'].shape[0]:
                    outputs_for_loss['text_embeddings'] = text_embeddings

                losses = self.loss_fn(outputs_for_loss)
                loss = losses['total']

            self.optim.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                self.optim.step()

            self.loss_history['steps'].append(step)
            self.loss_history['total'].append(float(loss))
            self.loss_history['reconstruction'].append(float(losses.get('reconstruction', torch.tensor(0.0))))
            self.loss_history['detail'].append(float(losses.get('detail', torch.tensor(0.0))))
            self.loss_history['uniformity'].append(float(losses.get('uniformity', torch.tensor(0.0))))
            self.loss_history['consistency'].append(float(losses.get('consistency', torch.tensor(0.0))))
            self.loss_history['clip'].append(float(losses.get('clip', torch.tensor(0.0))))
            
            recon_loss = float(losses.get('reconstruction', torch.tensor(0.0)))
            pbar.set_postfix({
                'recon_loss': f'{recon_loss:.4f}',
                'total_loss': f'{float(loss):.4f}'
            })
            
            if step % log_every == 0:
                recon = float(losses.get('reconstruction', torch.tensor(0.0)))
                detail = float(losses.get('detail', torch.tensor(0.0)))
                uni = float(losses.get('uniformity', torch.tensor(0.0)))
                cons = float(losses.get('consistency', torch.tensor(0.0)))
                clip = float(losses.get('clip', torch.tensor(0.0)))
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                remaining_steps = steps - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                print(f"\nstep {step:05d}/{steps:05d} ({step/steps*100:.1f}%) | "
                    f"total {float(loss):.4f} | recon {recon:.4f} | detail {detail:.4f} | uni {uni:.4f} | cons {cons:.4f} | clip {clip:.4f} | "
                        f"w_uni {self.loss_fn.uniformity_weight:.4f} | w_cons {self.loss_fn.consistency_weight:.4f} | "
                      f"ETA: {eta_hours:.2f}h ({steps_per_sec:.2f} steps/s)")
            
            if self.output_dir:
                self._save_checkpoint(step)
                self._save_loss_plots(step)
                
                # 更高频率地保存重建可视化，便于观察训练效果
                if step % 200 == 0 or step == steps:
                    self._save_reconstructions(step)
        
        pbar.close()
        total_time = time.time() - start_time
        total_hours = total_time / 3600
        print(f"\nTraining completed in {total_hours:.2f} hours ({total_time:.0f} seconds)")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        step = int(checkpoint.get('step', 0))
        self.start_step = step
        return step
    
    def _save_checkpoint(self, step: int):
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'model_config': {
                'input_sources': dict(getattr(self.model, 'input_sources', {})),
                'decode_sources': dict(getattr(self.model, 'decode_sources', {})),
            },
        }
        torch.save(checkpoint, output_path / 'checkpoint_latest.pt')
    
    def _save_reconstructions(self, step: int):
        output_path = Path(self.output_dir) / 'reconstructions'
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = self.dataloader.dataset
        collate_fn = getattr(self.dataloader, 'collate_fn', None)
        if collate_fn is None:
            return

        # 每次保存时为每个源随机抽取少量样本，而不是固定使用同一张影像
        # 注意控制样本数量，避免在大模型/长时间序列上导致显存占用过高。
        max_samples_per_source = 4
        for src_name in self.model.decode_sources.keys():
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            selected_samples: List[Dict[str, Any]] = []
            for idx in indices:
                sample = dataset[idx]
                frame_masks = sample.get('frame_valid_mask', {})
                has_valid = False
                if src_name in frame_masks and bool(frame_masks[src_name].any()):
                    has_valid = True
                elif src_name not in frame_masks and bool(sample['source_data'][src_name].abs().sum() > 0):
                    has_valid = True
                if has_valid:
                    selected_samples.append(sample)
                if len(selected_samples) >= max_samples_per_source:
                    break

            if not selected_samples:
                continue

            preview_batch = collate_fn(selected_samples)

            try:
                predictions, targets, masks = self._run_reconstruction_preview(preview_batch)
            except RuntimeError as e:
                # 若在可视化阶段出现 CUDA OOM，则跳过本次可视化以保证训练不中断。
                if 'out of memory' in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Warning: CUDA OOM during reconstruction preview for source '{src_name}' at step {step}; skipping visualization.")
                    continue
                raise
            if src_name not in predictions or src_name not in targets or src_name not in masks:
                continue

            pred = predictions[src_name]
            target = targets[src_name]
            mask = masks[src_name]

            valid_rows = [
                idx for idx in range(pred.shape[0])
                if mask[idx].sum().item() > 0 and target[idx].abs().sum().item() > 0
            ]
            if not valid_rows:
                continue
            
            # 每个源最多展示 8 个样本，以输出更多重建可视化图像
            display_rows = valid_rows[:8]
            B = len(display_rows)
            fig, axes = plt.subplots(B, 2, figsize=(8, 4 * B))
            axes = axes.reshape(B, 2) if B == 1 else axes
            
            for row_idx, batch_idx in enumerate(display_rows):
                pred_b = pred[batch_idx].numpy()
                target_b = target[batch_idx].numpy()

                pred_rgb = self._stretch_for_display(self._to_display_rgb(src_name, pred_b))
                target_rgb = self._stretch_for_display(self._to_display_rgb(src_name, target_b))
                
                axes[row_idx, 0].imshow(target_rgb)
                axes[row_idx, 0].set_title('Target RGB')
                axes[row_idx, 0].axis('off')
                
                axes[row_idx, 1].imshow(pred_rgb)
                axes[row_idx, 1].set_title('Prediction RGB')
                axes[row_idx, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{src_name}_step_{step:05d}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _save_loss_plots(self, step: int):
        """Save training loss plots focusing on reconstruction loss."""
        output_path = Path(self.output_dir) / 'plots'
        output_path.mkdir(parents=True, exist_ok=True)
        
        steps = np.array(self.loss_history['steps'])
        recon_loss = np.array(self.loss_history['reconstruction'])
        
        # Plot 1: Reconstruction loss
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(steps, recon_loss, label='Reconstruction Loss', linewidth=2, color='C0')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.savefig(output_path / 'reconstruction_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Smoothed reconstruction loss (moving average) for better visualization
        window_size = min(50, len(steps) // 10 + 1)
        
        def smooth(values, window):
            smoothed = []
            for i in range(len(values)):
                start = max(0, i - window // 2)
                end = min(len(values), i + window // 2 + 1)
                smoothed.append(np.mean(values[start:end]))
            return smoothed
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(steps, recon_loss, label='Reconstruction Loss (raw)', alpha=0.3, linewidth=1, color='C0')
        ax.plot(steps, smooth(recon_loss, window_size), 
               label=f'Reconstruction Loss (smoothed, window={window_size})', linewidth=2, color='C0')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Reconstruction Loss (Smoothed)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        plt.tight_layout()
        plt.savefig(output_path / 'reconstruction_loss_smoothed.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_trainer(model: AlphaEarthFoundations,
                   dataloader,
                   text_adapter = None,
                   lr: float = 1e-4,
                   device: Optional[str] = None,
                   output_dir: Optional[str] = None,
                   reconstruction_weight: float = 1.0,
                   uniformity_weight: float = 0.01,
                   consistency_weight: float = 0.005,
                   detail_weight: float = 0.05,
                   text_weight: float = 0.001,
                   use_amp: bool = True,
                   uniformity_ramp_steps: int = 0,
                   consistency_ramp_steps: int = 0) -> Trainer:
    return Trainer(
        model=model,
        dataloader=dataloader,
        text_adapter=text_adapter,
        lr=lr,
        device=device,
        output_dir=output_dir,
        reconstruction_weight=reconstruction_weight,
        uniformity_weight=uniformity_weight,
        consistency_weight=consistency_weight,
        detail_weight=detail_weight,
        text_weight=text_weight,
        use_amp=use_amp,
        uniformity_ramp_steps=uniformity_ramp_steps,
        consistency_ramp_steps=consistency_ramp_steps,
    )
