# classifier/model/model/clip.py
import torch.nn as nn
from model.model.cross_attention import CrossAttentionMaskFusion
import torch.nn.functional as F
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
import torch
try:
    import open_clip
    _HAVE_OPEN_CLIP = True
except Exception as _e:
    _HAVE_OPEN_CLIP = False
    _OPEN_CLIP_ERR = _e


class CLIPBuilder:
    @staticmethod
    def _build_clip_vit_b32(num_classes: int, pretrained: str = "openai", use_attn_mask: bool = False, use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        if not _HAVE_OPEN_CLIP:
            raise ImportError(
                "open_clip is required for CLIP. Install with `pip install open-clip-torch`.\n"
                f"Original error: {_OPEN_CLIP_ERR}"
            )

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=pretrained)
        visual = model.visual
        visual_dim = 768

        classifier = nn.Linear(768, num_classes)
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

        return CLIPWrapper(visual, classifier, cross_attn,spatial_attn, attn_pooling)


class CLIPWrapper(nn.Module):
    def __init__(self, visual, classifier, cross_attn, spatial_attn, attn_pooling):
        super().__init__()
        self.visual = visual
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling
    def _extract_spatial_tokens(self, x):
        # Convert to patch tokens
        x = self.visual.conv1(x)                          # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)         # [B, C, N]
        x = x.permute(0, 2, 1)                            # [B, N, C]
        x = torch.cat([self.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)  # Add CLS token

        x = x + self.visual.positional_embedding.to(x.dtype)  # [B, N+1, C]
        x = self.visual.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)                            # [N+1, B, C]
        x = self.visual.transformer(x)                    # [N+1, B, C]
        x = x.permute(1, 0, 2)                            # [B, N+1, C]

        return x  # [B, N+1, C] — includes CLS

    def forward(self, batch):
        x = batch["rgb"]
        mask = batch.get("masks", None)

        tokens = self._extract_spatial_tokens(x)      # [B, 197, D]
        patch_tokens = tokens[:, 1:, :]               # Drop CLS
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        feats = patch_tokens.transpose(1, 2).reshape(B, D, H, W)  # [B, D, H, W]
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        if self.cross_attn is not None and mask is not None:
            feats = self.cross_attn(feats, mask)
        if self.spatial_attn is not None:
            feats = self.spatial_attn(feats, mask)
        if self.attn_pooling is not None:
            pooled = self.attn_pooling(feats, mask)
        else:
            pooled = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        return self.classifier(pooled)