import torch
import torch.nn as nn

class ClassifierGlue(nn.Module):
    """Concatenate enabled feature blocks then classify."""
    def __init__(self, d_feat: int, d_rel: int, num_classes: int, linear_head: bool = False):
        super().__init__()
        in_dim = d_feat +  d_rel
        if linear_head:
            self.fc = nn.Linear(in_dim, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h = torch.cat(xs, dim=-1) if len(xs) > 1 else xs[0]
        return self.fc(x)
