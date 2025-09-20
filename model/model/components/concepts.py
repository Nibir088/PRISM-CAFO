import torch
import torch.nn as nn

class ConceptHead(nn.Module):
    """Tiny presence head: 1x1 conv -> GAP -> sigmoid."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.conv(feats)          # [B,1,H,W]
        x = x.mean(dim=(2, 3))        # [B,1]
        return torch.sigmoid(x)       # [B,1]
