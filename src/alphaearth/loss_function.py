
from typing import Any, Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.functional import F

"""
AEF loss = Reconstruction loss + Uniformity loss + Consistency loss + Text loss
"""

class AEFLoss:
    """
    AlphaEarth Foundations loss implementation following Equation 3 in the paper:
    """
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,  # a = 1.0
                 uniformity_weight: float = 0.01,    # lower regularization improves reconstruction convergence
                 consistency_weight: float = 0.005,  # lower regularization improves reconstruction convergence
                 text_weight: float = 0.001,
                 detail_weight: float = 0.05):       # edge/detail matching term
        
        self.reconstruction_weight = reconstruction_weight
        self.uniformity_weight = uniformity_weight
        self.consistency_weight = consistency_weight
        self.text_weight = text_weight
        self.detail_weight = detail_weight
        
        # Source-specific loss configurations (reconstruction term)
        # 参考你给出的原始实现：连续型源使用 L1 损失，并允许按源加权。
        # 这里为当前 S1/S2 训练场景设置默认权重，可按需要调整。
        self.source_configs = {
            'landsat':   {'weight': 1.0, 'loss_name': 'l1', 'beta': 0.05},
            # Slightly up-weight S2 so training focuses more on optical reconstruction fidelity.
            'sentinel2': {'weight': 1.5, 'loss_name': 'l1', 'beta': 0.05},
            # S1 噪声更大，这里进一步降低其在重建损失中的权重，
            # 让模型更关注 S2 的结构重建，减轻 S1 对整体梯度的干扰。
            'sentinel1': {'weight': 0.1, 'loss_name': 'l1', 'beta': 0.05},
        }

    def _masked_regression_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        loss_name: str,
        beta: float,
    ) -> torch.Tensor | None:
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)

        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(-1)

        if mask.shape != prediction.shape:
            if mask.shape[-1] == 1 and prediction.shape[-1] != 1:
                mask = mask.expand_as(prediction)
            else:
                mask = torch.broadcast_to(mask, prediction.shape)

        valid_weight = mask.sum()
        if valid_weight.item() <= 0:
            return None

        if loss_name == 'smooth_l1':
            per_element = nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction='none',
                beta=beta,
            )
        else:
            # 与你给出的原始版本保持一致：连续源采用 L1 损失
            per_element = nn.functional.l1_loss(
                prediction,
                target,
                reduction='none',
            )

        return (per_element * mask).sum() / valid_weight.clamp_min(1.0)
    
    def reconstruction_loss(self, predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss for all sources

        Compares predicted observation y_i' with ground truth y_i for each source i --> this leads the model to force the embeddings to carry enough information to be able to reconstruct the raw EO inputs. 
        For continuous sources: use L1 Loss; for categorical sources: use Cross-entropy.

        """
        
        total_loss = None
        
        for source in predictions:
            if source in targets:
                # 对未知 source，默认使用权重 1.0、L1 损失
                config = self.source_configs.get(source, {
                    'weight': 1.0,
                    'loss_name': 'l1',
                    'beta': 0.05,
                })
                prediction = predictions[source]
                target = targets[source]
                mask = masks.get(source, torch.ones_like(target[..., :1]))

                loss = self._masked_regression_loss(
                    prediction=prediction,
                    target=target,
                    mask=mask,
                    loss_name=config['loss_name'],
                    beta=config.get('beta', 0.05),
                )
                if loss is None:
                    continue

                # weight the loss by source
                weighted_loss = config['weight'] * loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        if total_loss is None:
            device = next(iter(predictions.values())).device if predictions else 'cpu'
            return torch.tensor(0.0, device=device)
        return total_loss

    def _masked_detail_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Gradient-domain L1 loss to recover sharper pixel-level structures.

        prediction/target: (B, H, W, C)
        mask: (B, H, W, 1) or broadcastable
        """
        mask = mask.to(device=prediction.device, dtype=prediction.dtype)

        while mask.dim() < prediction.dim():
            mask = mask.unsqueeze(-1)

        if mask.shape != prediction.shape:
            if mask.shape[-1] == 1 and prediction.shape[-1] != 1:
                mask = mask.expand_as(prediction)
            else:
                mask = torch.broadcast_to(mask, prediction.shape)

        # Finite-difference gradients
        pred_dx = prediction[:, 1:, :, :] - prediction[:, :-1, :, :]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        tgt_dx = target[:, 1:, :, :] - target[:, :-1, :, :]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Only count gradients where both adjacent pixels are valid
        mask_dx = mask[:, 1:, :, :] * mask[:, :-1, :, :]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        w_dx = mask_dx.sum()
        w_dy = mask_dy.sum()
        if (w_dx + w_dy).item() <= 0:
            return None

        loss_dx = ((pred_dx - tgt_dx).abs() * mask_dx).sum() / w_dx.clamp_min(1.0)
        loss_dy = ((pred_dy - tgt_dy).abs() * mask_dy).sum() / w_dy.clamp_min(1.0)
        return 0.5 * (loss_dx + loss_dy)

    def detail_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = None

        for source in predictions:
            if source not in targets:
                continue

            config = self.source_configs.get(source, {'weight': 1.0})
            prediction = predictions[source]
            target = targets[source]
            mask = masks.get(source, torch.ones_like(target[..., :1]))

            loss = self._masked_detail_loss(prediction, target, mask)
            if loss is None:
                continue

            weighted = config.get('weight', 1.0) * loss
            total = weighted if total is None else total + weighted

        if total is None:
            device = next(iter(predictions.values())).device if predictions else 'cpu'
            return torch.tensor(0.0, device=device)
        return total

    def batch_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute batch uniformity objective (Equation 4) --> objective: to have the embeddings be uniformly distributed.
        Takes the embeddings, rotates & shuffles them across the batch and then minimizes the absolute dot product between matched pairs
        """
        # embeddings: (B, H, W, D) or (B, T, H, W, D); flatten to N vectors in D
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        else:
            # (N, D)
            pass

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # Rotate (roll) sample pairs to approximate u' in the paper
        x_prime = torch.roll(x, shifts=1, dims=0)
        dots = (x * x_prime).sum(dim=-1).abs()  # |u · u'|
        return dots.mean()

    
    def consistency_loss(self, teacher_embeddings: torch.Tensor, 
                        student_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute teacher-student consistency loss (Equation 5)."""
        # 1 - mu · mu_s over 2, averaged over all pixels
        mu = torch.nn.functional.normalize(teacher_embeddings, p=2, dim=-1)
        mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
        dots = (mu * mu_s).sum(dim=-1)
        return ((1.0 - dots) * 0.5).mean()
    
    def clip_loss(self, image_embeddings: torch.Tensor, 
                  text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute CLIP-style contrastive loss."""
        # Expect (B, D) vs (B, D)
        img = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        txt = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        logits = img @ txt.t()  # (B, B)
        targets = torch.arange(img.size(0), device=img.device)
        loss_i = torch.nn.functional.cross_entropy(logits, targets)
        loss_t = torch.nn.functional.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_i + loss_t)  
    
    def __call__(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute total loss following Equation 3."""
        
        losses = {}
        
        if 'predictions' in outputs and 'targets' in outputs:
            recon_loss = self.reconstruction_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs.get('masks', {})
            )
            losses['reconstruction'] = recon_loss
            losses['detail'] = self.detail_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs.get('masks', {}),
            )
        else:
            losses['reconstruction'] = torch.tensor(0.0)
            losses['detail'] = torch.tensor(0.0)
        
        if 'embeddings' in outputs:
            uniformity_loss = self.batch_uniformity_loss(outputs['embeddings'])
            losses['uniformity'] = uniformity_loss
        else:
            losses['uniformity'] = torch.tensor(0.0, device=next(iter(outputs.values())).device if outputs else 'cpu')

        if 'teacher_embeddings' in outputs and 'student_embeddings' in outputs:
            consistency_loss = self.consistency_loss(
                outputs['teacher_embeddings'],
                outputs['student_embeddings']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        if 'image_embeddings' in outputs and 'text_embeddings' in outputs:
            clip_loss = self.clip_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            losses['clip'] = clip_loss
        else:
            losses['clip'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.detail_weight * losses['detail'] +
            self.uniformity_weight * losses['uniformity'] +
            self.consistency_weight * losses['consistency'] +
            self.text_weight * losses['clip']
        )
        
        losses['total'] = total_loss
        
        return losses
