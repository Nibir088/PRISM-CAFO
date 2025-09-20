# classifier/model/model/remoteclip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model.cross_attention import CrossAttentionMaskFusion
import os
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
try:
    import open_clip
    _HAVE_OPEN_CLIP = True
except Exception as _e:
    _HAVE_OPEN_CLIP = False
    _OPEN_CLIP_ERR = _e


class RemoteCLIPBuilder:
    @staticmethod
    def _build_remoteclip_vit_b32(num_classes: int, ckpt_path: str, use_attn_mask: bool = False, use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        if not _HAVE_OPEN_CLIP:
            raise ImportError(
                "open_clip is required for RemoteCLIP. Install with `pip install open-clip-torch`.\n"
                f"Original error: {_OPEN_CLIP_ERR}"
            )
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError("RemoteCLIP checkpoint not found. Provide `ckpt_path`.")

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

        visual = model.visual
        visual_dim = 768

        classifier = nn.Linear(visual_dim, num_classes)
        if use_attn_mask:
            cross_attn = CrossAttentionMaskFusion(visual_dim)
        else:
            cross_attn = None
        if use_spatial_attn:
            spatial_attn = SpatialAttention(visual_dim, 7)
        else:
            spatial_attn = None
        if use_attn_pooling:
            attn_pooling = MaskAttentionPooling(visual_dim, 7)
        else:
            attn_pooling = None

        return RemoteCLIPWrapper(visual, classifier, cross_attn,spatial_attn, attn_pooling)


import torch.nn.functional as F
import torch

class RemoteCLIPWrapper(nn.Module):
    def __init__(self, visual, classifier, cross_attn, spatial_attn, attn_pooling):
        super().__init__()
        self.visual = visual
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

    def _extract_spatial_tokens(self, x):
        # This replicates OpenCLIP's VisionTransformer forward
        x = self.visual.conv1(x)                                      # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)                     # [B, C, N]
        x = x.permute(0, 2, 1)                                        # [B, N, C]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x],
            dim=1
        )                                                             # [B, N+1, C]
        x = x + self.visual.positional_embedding.to(x.dtype)         # [B, N+1, C]
        x = self.visual.ln_pre(x)                                     # [B, N+1, C]

        x = x.permute(1, 0, 2)                                        # [N+1, B, C]
        x = self.visual.transformer(x)                                # [N+1, B, C]
        x = x.permute(1, 0, 2)                                        # [B, N+1, C]

        return x  # includes CLS token

    def forward(self, batch):
        x = batch["rgb"]
        mask = batch.get("masks", None)

        # Extract patch tokens
        tokens = self._extract_spatial_tokens(x)           # [B, N+1, D]
        patch_tokens = tokens[:, 1:, :]                    # Drop CLS token
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        feats = patch_tokens.transpose(1, 2).reshape(B, D, H, W)  # [B, D, H, W]

        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        # Step 3: Cross-attention
        if self.cross_attn is not None and mask is not None:
            feats = self.cross_attn(feats, mask)         # [B, D, H, W]
        if self.spatial_attn is not None:
            feats = self.spatial_attn(feats, mask)
        if self.attn_pooling is not None:
            pooled = self.attn_pooling(feats, mask)
        else:
            pooled = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return self.classifier(pooled)                     # [B, num_classes]
