import torch
import torch.nn as nn

class EvidenceHead(nn.Module):
    """Evidence map S = σ(Conv1x1(feat_map)), lazy to infer channels."""
    def __init__(self, visual_dim):
        super().__init__()
        self.head = nn.Conv2d(visual_dim, 1, kernel_size=1)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(feat_map))  # [B,1,H',W']
