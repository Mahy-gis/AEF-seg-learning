import torch
import torch.nn as nn

from .train_unet_from_embeddings import DoubleConv, compute_segmentation_metrics


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class LightDownsampleBlock(nn.Module):
    """Lightweight learnable downsampling using Conv(stride=2) + GN + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = _make_group_norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LightUNetDecoder(nn.Module):
    """UNet-style decoder over 4 pyramid scales (H, H/2, H/4, H/8)."""

    def __init__(self, base_ch: int, num_classes: int, norm: str = "group"):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, norm=norm)
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    @staticmethod
    def _match_spatial(skip: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if skip.shape[-2:] == gate.shape[-2:]:
            return skip, gate

        h = min(skip.shape[-2], gate.shape[-2])
        w = min(skip.shape[-1], gate.shape[-1])

        def center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            _, _, th, tw = t.shape
            start_h = max((th - target_h) // 2, 0)
            start_w = max((tw - target_w) // 2, 0)
            return t[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

        return center_crop(skip, h, w), center_crop(gate, h, w)

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up3(c4)
        c3, x = self._match_spatial(c3, x)
        x = torch.cat([c3, x], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        c2, x = self._match_spatial(c2, x)
        x = torch.cat([c2, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        c1, x = self._match_spatial(c1, x)
        x = torch.cat([c1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)


class UNetFromEmbeddings(nn.Module):
    """Embedding -> lightweight pyramid -> UNet decoder -> segmentation.

    The pyramid uses lightweight learnable downsampling to preserve the
    embedding semantics while learning multi-scale structure and class mapping.
    """

    def __init__(
        self,
        embedding_channels: int,
        num_classes: int = 9,
        base_ch: int = 8,
        norm: str = "group",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.embed_proj = nn.Sequential(
            nn.Conv2d(embedding_channels, base_ch, kernel_size=1, bias=False),
            _make_group_norm(base_ch),
            nn.GELU(),
        )

        self.down1 = LightDownsampleBlock(base_ch, base_ch * 2)
        self.down2 = LightDownsampleBlock(base_ch * 2, base_ch * 4)
        self.down3 = LightDownsampleBlock(base_ch * 4, base_ch * 8)

        self.decoder = LightUNetDecoder(base_ch=base_ch, num_classes=num_classes, norm=norm)

        if freeze_encoder:
            for p in self.embed_proj.parameters():
                p.requires_grad = False

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # H, H/2, H/4, H/8 pyramid
        c1 = self.embed_proj(emb)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)

        logits = self.decoder(c1, c2, c3, c4)
        if logits.shape[-2:] != emb.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=emb.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def evaluate_model_on_loader(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
):
    """Evaluate `model` on `data_loader` and return average pixel_acc and mean IoU."""
    model.eval()
    pixel_accs = []
    mious = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            acc, miou = compute_segmentation_metrics(logits, y, num_classes=num_classes, ignore_index=ignore_index)
            pixel_accs.append(acc)
            mious.append(miou)

    if pixel_accs:
        return float(sum(pixel_accs) / len(pixel_accs)), float(sum(mious) / len(mious))
    return 0.0, 0.0
