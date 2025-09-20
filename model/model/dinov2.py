# classifier/model/model/dinov2.py
import torch.nn as nn
import timm
from model.model.cross_attention import CrossAttentionMaskFusion
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
import torch.nn.functional as F
class DinoV2Builder:
    @staticmethod
    def _build_dinov2_vit_b(num_classes: int, pretrained: bool = True, use_attn_mask: bool = False,use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        model = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained)
        visual_dim = model.num_features if hasattr(model, "num_features") else model.head.in_features

        model.head = nn.Identity()
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
        return DinoV2Wrapper(model, classifier, cross_attn, spatial_attn, attn_pooling)

import torch.nn.functional as F

class DinoV2Wrapper(nn.Module):
    def __init__(self, visual, classifier, cross_attn, spatial_attn, attn_pooling):
        super().__init__()
        self.visual = visual
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

    def forward(self, batch):
        x = batch["rgb"]
        mask = batch.get("masks", None)

        # Step 1: Extract patch tokens (with CLS)
        tokens = self.visual.forward_features(x)         # [B, N+1, D]
        patch_tokens = tokens[:, 1:, :]                  # remove CLS token

        # Step 2: Reshape to [B, D, H, W]
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        feats = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        
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
            
        return self.classifier(pooled)