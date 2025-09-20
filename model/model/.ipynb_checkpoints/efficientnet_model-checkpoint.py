# classifier/model/model/swin.py
import torch.nn as nn
from torchvision import models
from model.model.cross_attention import CrossAttentionMaskFusion
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
import torch.nn.functional as F
class EfficientNetBuilder:
    @staticmethod
    def _build_efficientnet(name: str, weights_enum, num_classes: int, use_attn_mask: bool = False, use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        m = getattr(models, name)(weights=weights_enum)
        visual_dim = m.classifier[1].in_features

        backbone = nn.Sequential(*list(m.children())[:-1])  # [B, C, H, W]
        pool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = m.classifier[1]
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

        return EfficientNetWrapper(backbone, pool, classifier, cross_attn,spatial_attn, attn_pooling)


class EfficientNetWrapper(nn.Module):
    def __init__(self, backbone, pool, classifier, cross_attn, spatial_attn, attn_pooling):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

    def forward(self, batch):
        x = batch["rgb"]
        mask = batch.get("masks", None)

        feats = self.backbone(x)
        _,_,H,W = feats.shape
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        if self.cross_attn is not None and mask is not None:
            feats = self.cross_attn(feats, mask)
        if self.spatial_attn is not None:
            feats = self.spatial_attn(feats, mask)
        if self.attn_pooling is not None:
            pooled = self.attn_pooling(feats, mask)
        else:
            pooled = self.pool(feats).squeeze(-1).squeeze(-1)
        return self.classifier(pooled)