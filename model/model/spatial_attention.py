
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention over feature maps, guided by mask maps.
    """
    def __init__(self, visual_dim: int, mask_channels: int = 7, attn_dim: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Conv2d(visual_dim, attn_dim, kernel_size=1)
        self.key_proj   = nn.Conv2d(mask_channels, attn_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(mask_channels, visual_dim, kernel_size=1)

        # Reduced output projection: bottleneck style
        self.out_proj = nn.Sequential(
            nn.Conv2d(visual_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, visual_dim, kernel_size=1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            visual_feats: [B, C, H, W]
            mask:         [B, 7, H, W]
        Output:
            fused_feats:  [B, C, H, W]
        """
        B, C, H, W = visual_feats.shape

        Q = self.query_proj(visual_feats)  # [B, attn_dim, H, W]
        K = self.key_proj(mask)            # [B, attn_dim, H, W]
        V = self.value_proj(mask)          # [B, C, H, W]

        # Flatten spatial dims
        Q_flat = Q.view(B, Q.shape[1], -1)         # [B, attn_dim, HW]
        K_flat = K.view(B, K.shape[1], -1)         # [B, attn_dim, HW]
        V_flat = V.view(B, C, -1)                  # [B, C, HW]

        # Attention scores: [B, HW, HW]
        attn_scores = torch.bmm(Q_flat.transpose(1, 2), K_flat)  # [B, HW, HW]
        attn_scores = self.softmax(attn_scores / (Q_flat.shape[1] ** 0.5))  # softmax over last dim

        # Apply attention
        out_flat = torch.bmm(V_flat, attn_scores)   # [B, C, HW]
        out = out_flat.view(B, C, H, W)             # [B, C, H, W]

        # Project + residual
        out = self.out_proj(out)
        return out + visual_feats
