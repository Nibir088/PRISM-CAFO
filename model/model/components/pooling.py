from typing import Tuple
import torch
import torch.nn as nn

class ComponentPooling(nn.Module):
    """Mask-guided soft ROI pooling + DeepSets aggregation."""
    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    @staticmethod
    def soft_roi_pool(feats: torch.Tensor, M: torch.Tensor, eps=1e-6) -> torch.Tensor:
        # feats:[B,D,H,W], M:[B,K,H,W] in [0,1]
        num = num = torch.einsum('bchw,bkhw->bkc', feats, M)                  # [B,K,C]
        den = M.flatten(2).sum(-1, keepdim=True).clamp_min(eps)          # [B,K,1]
        return num / den

    def forward(self, feats: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = self.soft_roi_pool(feats, M)         # [B,K,C]
        U = self.phi(Z)                          # [B,K,H]
        agg = self.rho(U.sum(dim=1))             # [B,d_out]
        return Z, agg


def gap_conf(feats: torch.Tensor, C: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Confidence-weighted GAP: Σ(feats·C)/Σ(C)."""
    num = (feats * C).sum(dim=(2, 3))
    den = C.sum(dim=(2, 3)).clamp_min(eps)
    return num / den


class GAPConf(nn.Module):
    def __init__(self, k_in: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(k_in, 1, kernel_size=3, padding=1, bias=True)  # keeps HxW

    def forward(self, feats: torch.Tensor, C: torch.Tensor, eps: float = 1e-6):
        # feats: [B,D,H,W], C: [B,K,H,W]
        W = torch.sigmoid(self.conv(C))               # [B,1,H,W], same size as input
        num = (feats * W).sum(dim=(2, 3))             # [B,D]
        den = W.sum(dim=(2, 3)).clamp_min(eps)        # [B,1]
        return num / den                              # [B,D]

