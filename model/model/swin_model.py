# classifier/model/model/efficientnet.py
import torch.nn as nn
from torchvision import models
from model.model.cross_attention import CrossAttentionMaskFusion
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
import torch.nn.functional as F
class SwinBuilder:
    @staticmethod
    def _build_swin_b(num_classes: int, pretrained: bool = True, use_attn_mask: bool = False, use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        swin = models.swin_b(weights=weights)

        visual_dim = swin.head.in_features
        swin.head = nn.Identity()
        pool = nn.AdaptiveAvgPool2d(1)
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
        classifier = nn.Linear(visual_dim, num_classes)

        return SwinWrapper(swin, pool, classifier, cross_attn, spatial_attn, attn_pooling)


class SwinWrapper(nn.Module):
    def __init__(self, swin, pool, classifier, cross_attn, spatial_attn, attn_pooling):
        super().__init__()
        self.swin = swin
        self.pool = pool
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

    def forward(self, batch):
        x = batch["rgb"]
        mask = batch.get("masks", None)

        feats = self.swin.features(x)         # [B, C, H, W]
        feats = feats.permute(0, 3, 1, 2)  # [B, C, H, W]
        # print(feats.shape)
        # Optional: mask-guided cross-attention
        _, _, H, W = feats.shape
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        if self.cross_attn is not None and mask is not None:
            feats = self.cross_attn(feats, mask)
            # print(feats.shape)
        if self.spatial_attn is not None:
            feats = self.spatial_attn(feats, mask)
        if self.attn_pooling is not None:
            pooled = self.attn_pooling(feats, mask)
        else:
            pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # [B, D]
        # print(pooled.shape)
        return self.classifier(pooled)
