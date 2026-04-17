import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# 复用已有的 embedding 数据集与工具
from seg.train_unet_from_embeddings import (
    EmbeddingSegmentationDataset,
    AugmentedTrainDataset,
    FeatureNormalizeDataset,
    estimate_feature_channel_stats,
    compute_segmentation_metrics,
    evaluate_per_class_iou,
    save_val_visualizations,
)


# -------------------- 模型组件：Feature Adapter + Pixel Decoder + Transformer Head --------------------


class AEFFeatureAdapter(nn.Module):
    """将 AEF 单尺度 64 维特征图转换为多尺度特征.

    设计参考你提供的 AEFFeatureAdapter：
      输入: [B, 64, 128, 128]
      输出: [P2, P3, P4]
        - P2: [B, C_out, 128, 128]
        - P3: [B, C_out, 64,  64]
        - P4: [B, C_out, 32,  32]
    """

    def __init__(self, in_dim: int = 64, out_dim: int = 256):
        super().__init__()
        # P2: 保持高分辨率，先升维再细化
        self.p2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        # P3: 下采样一倍
        self.p3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        # P4: 再下采样
        self.p4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 64, H, W]，通常 H=W=128
        p2 = self.p2(x)   # [B, C_out, H,   W]
        p3 = self.p3(p2)  # [B, C_out, H/2, W/2]
        p4 = self.p4(p3)  # [B, C_out, H/4, W/4]
        return [p2, p3, p4]


class PixelDecoderLite(nn.Module):
    """简化版 FPN 式 pixel decoder.

    - 输入: [P2, P3, P4]，每个 C_in 通道
    - 先用 1x1 lateral conv 映射到 out_channels
    - 自顶向下上采样+相加，再用 3x3 conv 输出
    - 返回最高分辨率的特征图 P2_fpn，作为 pixel-level memory
    """

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) for _ in range(num_levels)]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(8, out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels)
            ]
        )

    def forward(self, feats):
        # feats: [P2, P3, P4]  分辨率从高到低
        assert len(feats) == self.num_levels
        fpn_feats = []
        prev = None
        # 自顶向下：从 P4 -> P3 -> P2
        for i in range(self.num_levels - 1, -1, -1):
            lat = self.lateral_convs[i](feats[i])
            if prev is not None:
                lat = lat + F.interpolate(prev, size=lat.shape[2:], mode="bilinear", align_corners=False)
            out = self.output_convs[i](lat)
            fpn_feats.insert(0, out)
            prev = out

        # 返回最高分辨率的 P2 作为 pixel-level memory
        return fpn_feats[0]


def build_2d_sincos_position_embedding(h: int, w: int, dim: int, device) -> torch.Tensor:
    """标准 2D 正余弦位置编码，返回 (H*W, dim)."""

    if dim % 4 != 0:
        raise ValueError("Position embedding dimension must be divisible by 4")

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    out_y = torch.einsum("hw,c->hwc", y.float(), omega)
    out_x = torch.einsum("hw,c->hwc", x.float(), omega)

    pos = torch.cat(
        [torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1
    )  # (H,W,dim)
    return pos.view(-1, dim)


class Mask2FormerLite(nn.Module):
    """轻量版 Mask2Former 风格 head：

    - FeatureAdapter: 64 维 embedding -> 多尺度特征
    - PixelDecoderLite: 多尺度 -> 统一分辨率特征图
    - Transformer decoder + queries: 生成一组查询 embedding
    - 将查询 embedding 与像素特征做点积，得到每个 query 的 mask logit
    - 将 query 的类别 logit 与 mask logit 组合，得到像素级类别 logit

    最终输出标准语义分割 logits: (B, num_classes, H, W)
    """

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 128,
        num_queries: int = 64,
        num_classes: int = 3,
        num_decoder_layers: int = 4,
        feat_channels: int = 128,  # 兼容旧的构造参数，不实际使用
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        # Adapter 输出通道设置为 2*d_model，对齐你提供的 LightMask2Former 设计
        adapter_out_channels = d_model * 2
        self.adapter = AEFFeatureAdapter(in_dim=in_channels, out_dim=adapter_out_channels)
        self.pixel_decoder = PixelDecoderLite(in_channels=adapter_out_channels, out_channels=d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes)
        self.mask_embed = nn.Linear(d_model, d_model)
        self.mask_feat_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        B, _, H, W = x.shape
        feats = self.adapter(x)
        pixel_feats = self.pixel_decoder(feats)  # (B, d_model, H2, W2)
        B, C, H2, W2 = pixel_feats.shape

        # 位置编码 + transformer decoder
        pos = build_2d_sincos_position_embedding(H2, W2, C, device=x.device)  # (H2*W2, C)
        pixel_feats_flat = pixel_feats.flatten(2).permute(2, 0, 1)  # (H2*W2, B, C)
        pixel_feats_flat = pixel_feats_flat + pos[:, None, :]

        query = self.query_embed.weight  # (num_queries, C)
        query = query[:, None, :].expand(-1, B, -1)  # (num_queries, B, C)

        dec_out = self.decoder(tgt=query, memory=pixel_feats_flat)  # (num_queries, B, C)
        dec_out = dec_out.permute(1, 0, 2)  # (B, num_queries, C)

        class_logits = self.class_embed(dec_out)  # (B, Q, K)
        mask_embed = self.mask_embed(dec_out)     # (B, Q, C)

        mask_feats = self.mask_feat_proj(pixel_feats)  # (B, C, H2, W2)
        mask_feats_flat = mask_feats.view(B, C, H2 * W2)  # (B, C, HW)

        # 每个 query 与每个像素点点积 -> (B, Q, HW)
        mask_logits = torch.einsum("bqc,bch->bqh", mask_embed, mask_feats_flat)
        mask_logits = mask_logits.view(B, self.num_queries, H2 * W2)

        # 将 query 级别的类别 + mask 组合成像素级 logits
        # 对于每个类别 c，pixel_logit_c(h,w) = logsumexp_q ( log_softmax(class_logits[q,c]) + mask_logits[q,h,w] )
        cls_logit = nn.functional.log_softmax(class_logits, dim=-1)  # (B, Q, K)
        cls_logit = cls_logit.permute(0, 2, 1).unsqueeze(-1)         # (B, K, Q, 1)
        mask_logits = mask_logits.unsqueeze(1)                       # (B, 1, Q, HW)

        combined = cls_logit + mask_logits                           # (B, K, Q, HW)
        pixel_logits = torch.logsumexp(combined, dim=2)              # (B, K, HW)
        pixel_logits = pixel_logits.view(B, self.num_classes, H2, W2)

        # 上采样回原始分辨率
        out = nn.functional.interpolate(pixel_logits, size=(H, W), mode="bilinear", align_corners=False)
        return out


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        eps: float = 1e-6,
        background_index: int = 0,
        ignore_background: bool = True,
        focus_classes: Optional[tuple[int, ...]] = None,
        class_weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.background_index = background_index
        self.ignore_background = ignore_background
        self.focus_classes = focus_classes

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() < num_classes:
                w = torch.cat([w, torch.ones(num_classes - w.numel(), dtype=torch.float32)])
            elif w.numel() > num_classes:
                w = w[:num_classes]
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
        valid_mask = (target != self.ignore_index).float()  # (B,H,W)

        losses = []
        if self.focus_classes is None:
            iter_classes = range(self.num_classes)
        else:
            iter_classes = self.focus_classes

        for cls in iter_classes:
            if cls == self.ignore_index:
                continue
            if self.ignore_background and cls == self.background_index:
                continue

            tgt = (target == cls).float()
            if tgt.sum().item() <= 0:
                continue
            p = probs[:, cls, :, :] * valid_mask
            t = tgt * valid_mask
            inter = (p * t).sum()
            denom = (p * p).sum() + (t * t).sum()
            dice = (2.0 * inter + self.eps) / (denom + self.eps)
            loss_c = 1.0 - dice
            if self.class_weights is not None:
                loss_c = loss_c * self.class_weights[cls]
            losses.append(loss_c)

        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()


def foreground_class_aux_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_index: int,
    ignore_index: int,
) -> torch.Tensor:
    """简单的前景辅助损失：专门监督某个类别（如 1）vs 其他。

    - logits: (B,C,H,W)
    - target: (B,H,W)
    """

    probs = torch.softmax(logits, dim=1)[:, class_index, :, :]  # (B,H,W)
    valid = target != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)

    tgt = (target == class_index) & valid
    preds = probs[valid]
    tgt_f = tgt[valid].float()
    if tgt_f.numel() == 0:
        return logits.new_tensor(0.0)
    return F.binary_cross_entropy(preds.clamp(1e-6, 1.0 - 1e-6), tgt_f)


# -------------------- 训练脚本 --------------------


def build_datasets(
    embeddings_path: Path,
    labels_path: Path,
    val_embeddings_path: Optional[Path],
    val_labels_path: Optional[Path],
    val_fraction: float,
    embedding_key: str,
) -> Tuple[Dataset, Optional[Dataset]]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels path not found: {labels_path}")

    per_patch_labels_train = bool(labels_path.is_dir())
    full_dataset = EmbeddingSegmentationDataset(
        embeddings_path,
        labels_path,
        per_patch_labels=per_patch_labels_train,
        embedding_key=embedding_key,
    )

    if val_embeddings_path is not None and val_labels_path is not None:
        per_patch_labels_val = bool(val_labels_path.is_dir())
        val_full_dataset = EmbeddingSegmentationDataset(
            val_embeddings_path,
            val_labels_path,
            per_patch_labels=per_patch_labels_val,
            embedding_key=embedding_key,
        )
        return full_dataset, val_full_dataset

    # 单一集合上按比例划分
    n_total = len(full_dataset)
    if n_total <= 1 or val_fraction <= 0:
        return full_dataset, None

    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist())
    return train_ds, val_ds


def train(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    emb_path = Path(args.embeddings_path)
    labels_path = Path(args.labels_file)
    val_emb_path = Path(args.val_embeddings_path) if args.val_embeddings_path else None
    val_labels_path = Path(args.val_labels_file) if args.val_labels_file else None

    base_train_ds, base_val_ds = build_datasets(
        emb_path,
        labels_path,
        val_emb_path,
        val_labels_path,
        val_fraction=args.val_fraction,
        embedding_key=args.embedding_key,
    )

    # 推断通道数
    sample_feats, sample_labels = base_train_ds[0]
    C, H, W = sample_feats.shape
    print(f"Sample features shape: (C,H,W)=({C},{H},{W}), labels shape=({sample_labels.shape[0]},{sample_labels.shape[1]})")

    # 特征归一化
    if args.normalize_features == 1 and len(base_train_ds) > 0:
        feat_mean, feat_std = estimate_feature_channel_stats(base_train_ds, max_samples=args.feature_stats_max_samples)
        print(
            "Feature normalization: ON "
            f"(mean range [{feat_mean.min().item():.4f}, {feat_mean.max().item():.4f}], "
            f"std range [{feat_std.min().item():.4f}, {feat_std.max().item():.4f}])"
        )
        base_train_ds = FeatureNormalizeDataset(base_train_ds, feat_mean, feat_std)
        if base_val_ds is not None:
            base_val_ds = FeatureNormalizeDataset(base_val_ds, feat_mean, feat_std)
    else:
        print("Feature normalization: OFF")

    train_ds: Dataset = base_train_ds
    if args.enable_augmentation == 1 and len(base_train_ds) > 0:
        train_ds = AugmentedTrainDataset(
            base_dataset=base_train_ds,
            repeat_factor=args.train_repeat_factor,
            flip_prob=args.aug_flip_prob,
            rot90_prob=args.aug_rot90_prob,
            noise_std=args.aug_noise_std,
            gain_std=args.aug_gain_std,
            crop_size=args.train_crop_size,
            foreground_crop_prob=args.fg_crop_prob,
            background_index=args.background_index,
            ignore_index=args.ignore_index,
        )
        print("Train augmentation: ON")
    else:
        print("Train augmentation: OFF")

    val_ds = base_val_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    model = Mask2FormerLite(
        in_channels=C,
        feat_channels=args.feat_channels,
        d_model=args.d_model,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        num_decoder_layers=args.num_decoder_layers,
    )
    model.to(device)
    print(
        f"Model: Mask2FormerLite(in_channels={C}, feat_channels={args.feat_channels}, "
        f"d_model={args.d_model}, num_queries={args.num_queries}, num_classes={args.num_classes})"
    )

    # 损失：加权 CE + 仅关注前景类(通常是 1) 的 Dice
    ce_class_weights = None
    if args.class_weights:
        try:
            vals = [float(x) for x in args.class_weights.split(",") if x.strip() != ""]
            if vals:
                w = torch.as_tensor(vals, dtype=torch.float32)
                if w.numel() < args.num_classes:
                    w = torch.cat([w, torch.ones(args.num_classes - w.numel(), dtype=torch.float32)])
                elif w.numel() > args.num_classes:
                    w = w[: args.num_classes]
                ce_class_weights = w
                print(f"Using CE class weights: {ce_class_weights.tolist()}")
        except ValueError:
            print(f"Warning: failed to parse --class_weights='{args.class_weights}', using uniform weights.")

    criterion_ce = nn.CrossEntropyLoss(
        weight=ce_class_weights.to(device) if ce_class_weights is not None else None,
        ignore_index=args.ignore_index,
    )
    criterion_dice = SoftDiceLoss(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        background_index=args.background_index,
        ignore_background=True,
        # 同时关注类别 1 和 2，并对 1 给予更高权重
        focus_classes=(1, 2),
        class_weights=[0.0, 2.0, 1.0],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if args.scheduler_monitor == "val_miou" else "min",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_val_miou = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        running_miou_sum = 0.0
        running_miou_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss_ce = criterion_ce(logits, y)
            loss_dice = criterion_dice(logits, y)
            loss_aux = foreground_class_aux_loss(
                logits,
                y,
                class_index=args.focus_class,
                ignore_index=args.ignore_index,
            )
            loss = loss_ce + args.dice_weight * loss_dice + args.fg_aux_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            with torch.no_grad():
                acc, miou = compute_segmentation_metrics(
                    logits,
                    y,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index,
                    background_index=args.background_index,
                    ignore_background=True,
                )
                mask = y != args.ignore_index
                total = mask.sum().item()
                running_correct += int(round(acc * total))
                running_total += total
                running_miou_sum += miou
                running_miou_count += 1

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = running_correct / max(1, running_total) if running_total > 0 else 0.0
        train_miou = running_miou_sum / max(1, running_miou_count)

        val_loss = None
        val_acc = None
        val_miou = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_correct = 0
            val_running_total = 0
            val_running_miou_sum = 0.0
            val_running_miou_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    l_ce = criterion_ce(logits, y)
                    l_dice = criterion_dice(logits, y)
                    l_aux = foreground_class_aux_loss(
                        logits,
                        y,
                        class_index=args.focus_class,
                        ignore_index=args.ignore_index,
                    )
                    l = l_ce + args.dice_weight * l_dice + args.fg_aux_weight * l_aux
                    val_running_loss += l.item() * x.size(0)

                    acc, miou = compute_segmentation_metrics(
                        logits,
                        y,
                        num_classes=args.num_classes,
                        ignore_index=args.ignore_index,
                        background_index=args.background_index,
                        ignore_background=True,
                    )
                    mask = y != args.ignore_index
                    total = mask.sum().item()
                    val_running_correct += int(round(acc * total))
                    val_running_total += total
                    val_running_miou_sum += miou
                    val_running_miou_count += 1

            val_loss = val_running_loss / max(1, len(val_loader.dataset))
            val_acc = val_running_correct / max(1, val_running_total) if val_running_total > 0 else 0.0
            val_miou = val_running_miou_sum / max(1, val_running_miou_count)

            improved = val_miou > best_val_miou
            if improved:
                best_val_miou = val_miou
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if val_loss is not None:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mIoU: {val_miou:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )
            if args.scheduler_monitor == "val_miou":
                scheduler.step(val_miou)
            else:
                scheduler.step(val_loss)
        else:
            print(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_mIoU: {train_miou:.4f}, "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )
            scheduler.step(train_loss)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch+1} (no val mIoU improvement for {args.early_stop_patience} epochs)."
            )
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_latest = out_dir / "mask2former_from_embeddings_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "in_channels": C}, ckpt_latest)
    print(f"Saved latest checkpoint to {ckpt_latest}")

    if best_state is not None:
        ckpt_best = out_dir / "mask2former_from_embeddings_best.pt"
        torch.save({"model_state_dict": best_state, "in_channels": C}, ckpt_best)
        print(f"Saved best checkpoint to {ckpt_best} (best val mIoU={best_val_miou:.4f})")

        if val_loader is not None and val_ds is not None and len(val_ds) > 0:
            model.load_state_dict(best_state)
            per_class_iou = evaluate_per_class_iou(
                model=model,
                data_loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                background_index=args.background_index,
                ignore_background=True,
            )
            print("Per-class IoU on best checkpoint (NaN = ignored / no pixels):")
            for cls_idx, iou in enumerate(per_class_iou):
                print(
                    f"  class {cls_idx}: {iou:.4f}" if not np.isnan(iou) else f"  class {cls_idx}: NaN"
                )
            np.save(out_dir / "best_val_per_class_iou.npy", per_class_iou)

    save_val_visualizations(
        model=model,
        val_ds=val_ds,
        device=device,
        output_dir=out_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.save_val_samples,
    )

    # 返回本次训练的关键验证指标，便于在外部脚本中做超参搜索对比。
    # 若没有验证集，则 best_val_miou 可能始终为 0，仅供参考。
    return {
        "best_val_miou": float(best_val_miou),
        # 目前未显式保存 best_val_loss/best_val_acc，可在需要时进一步扩展。
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a lightweight Mask2Former-style segmentation head using AEF embeddings "
            "as input features and integer label mask as targets."
        )
    )
    p.add_argument("--embeddings_path", type=str, required=True)
    p.add_argument("--embedding_key", type=str, default="embeddings_native")
    p.add_argument("--labels_file", type=str, required=True)
    p.add_argument("--val_embeddings_path", type=str, default="")
    p.add_argument("--val_labels_file", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--ignore_index", type=int, default=255)
    p.add_argument("--background_index", type=int, default=0)
    p.add_argument("--focus_class", type=int, default=1, help="前景主类索引（通常为 1）")

    p.add_argument("--feat_channels", type=int, default=128)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_queries", type=int, default=64)
    p.add_argument("--num_decoder_layers", type=int, default=4)

    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--enable_augmentation", type=int, default=1, choices=[0, 1])
    p.add_argument("--normalize_features", type=int, default=1, choices=[0, 1])
    p.add_argument("--feature_stats_max_samples", type=int, default=0)

    p.add_argument("--train_repeat_factor", type=int, default=4)
    p.add_argument("--aug_flip_prob", type=float, default=0.5)
    p.add_argument("--aug_rot90_prob", type=float, default=0.5)
    p.add_argument("--aug_noise_std", type=float, default=0.01)
    p.add_argument("--aug_gain_std", type=float, default=0.05)
    p.add_argument("--train_crop_size", type=int, default=96)
    p.add_argument("--fg_crop_prob", type=float, default=0.5)

    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--class_weights", type=str, default="1,6,1", help="CE 类别权重，如 '1,6,1'")
    p.add_argument(
        "--fg_aux_weight",
        type=float,
        default=0.4,
        help="前景辅助损失（focus_class vs 其他）的权重",
    )

    p.add_argument("--lr_patience", type=int, default=8)
    p.add_argument("--scheduler_monitor", type=str, default="val_miou", choices=["val_loss", "val_miou"])
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--early_stop_patience", type=int, default=30)

    p.add_argument("--save_val_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
