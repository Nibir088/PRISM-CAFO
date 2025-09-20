import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskAttentionPooling(nn.Module):
    def __init__(self, visual_dim: int, mask_channels: int = 7, attn_hidden: int = 64):
        super().__init__()
        # Predict spatial attention scores from the mask
        self.mask_attn = nn.Sequential(
            nn.Conv2d(mask_channels, attn_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_hidden, 1, kernel_size=1),   # [B, 1, H, W]
            nn.Sigmoid()
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, D, H, W] — visual features
            mask:  [B, 7, H, W] — soft masks (e.g. per component type)

        Returns:
            pooled_feats: [B, D] — attention-weighted pooled visual features
        """
        attn_map = self.mask_attn(mask)               # [B, 1, H, W]
        weighted_feats = feats * attn_map             # [B, D, H, W]

        # Global average pooling over spatial dims, weighted by attn
        pooled_feats = weighted_feats.flatten(2).sum(dim=2) / (attn_map.flatten(2).sum(dim=2) + 1e-6)  # [B, D]
        return pooled_feats
