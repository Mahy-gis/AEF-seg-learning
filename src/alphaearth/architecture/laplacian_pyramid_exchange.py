from torch import nn
import torch  
import torch.nn.functional as F

class LearnedSpatialResampling(nn.Module):
    """Learned spatial rescaling with anti-artifact interpolation + convolution.

    Previous ConvTranspose2d upsampling could introduce checkerboard/stripe
    artifacts. This module uses explicit interpolation followed by Conv2d,
    which is more stable for reconstruction quality.
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

        # Always use conv after resize. Keep kernel small and symmetric to
        # reduce directional artifacts.
        if scale_factor == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1:
            return self.conv(x)

        # Use interpolation for deterministic resize and then refine by conv.
        mode = 'bilinear'
        align_corners = False
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=False,
        )
        return self.conv(x)
