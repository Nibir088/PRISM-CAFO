# classifier/model/cross_attention.py
import torch
import torch.nn as nn

class CrossAttentionMaskFusion(nn.Module):
    def __init__(self, visual_dim: int, mask_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(7, mask_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
        )
        self.query_proj = nn.Linear(visual_dim, visual_dim)
        self.key_proj = nn.Linear(mask_dim, visual_dim)
        self.value_proj = nn.Linear(mask_dim, visual_dim)
        self.attn = nn.MultiheadAttention(visual_dim, num_heads=n_heads, batch_first=True)

    def forward(self, visual_feats, mask):
        B = visual_feats.size(0)
        C, H, W = visual_feats.shape[1:]
        q = visual_feats.flatten(2).permute(0, 2, 1)       # [B, HW, C]

        mask_feats = self.mask_encoder(mask)               # [B, Dm, H, W]
        k = mask_feats.flatten(2).permute(0, 2, 1)         # [B, HW, Dm]
        k = self.key_proj(k)
        v = self.value_proj(mask_feats.flatten(2).permute(0, 2, 1))

        q_proj = self.query_proj(q)
        out, _ = self.attn(q_proj, k, v)
        return out.permute(0, 2, 1).view(B, C, H, W)
