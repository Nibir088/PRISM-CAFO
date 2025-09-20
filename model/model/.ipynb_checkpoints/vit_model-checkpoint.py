# classifier/model/model/vit.py
import torch.nn as nn
from torchvision import models
from model.model.cross_attention import CrossAttentionMaskFusion
import torch
import torch.nn.functional as F
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
class ViTBuilder:
    @staticmethod
    def _build_vit_b_16(num_classes: int, pretrained: bool = True, use_attn_mask: bool = False, use_spatial_attn: bool = True, use_attn_pooling:bool=True) -> nn.Module:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = models.vit_b_16(weights=weights)
        visual_dim = vit.heads[-1].in_features if isinstance(vit.heads, nn.Sequential) else vit.heads.in_features

        vit.heads = nn.Identity()
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

        return ViTWrapper(vit, classifier, cross_attn,spatial_attn, attn_pooling)


import torch.nn.functional as F

class ViTWrapper(nn.Module):
    def __init__(self, vit, classifier, cross_attn,spatial_attn, attn_pooling):
        super().__init__()
        self.vit = vit
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

    def forward(self, batch):
        x = batch["rgb"]                     # [B, 3, H, W]
        mask = batch.get("masks", None)

        B = x.size(0)

        # Patch embedding
        x = self.vit._process_input(x)       # [B, N, D] = [B, 196, D]

        # Add CLS token
        cls_token = self.vit.class_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_token, x), dim=1)              # [B, 197, D]

        # Add positional encoding
        x = x + self.vit.encoder.pos_embedding            # [B, 197, D]
        x = self.vit.encoder.dropout(x)

        # Transformer blocks
        x = self.vit.encoder.layers(x)                    # [B, 197, D]
        x = self.vit.encoder.ln(x)

        # Now drop CLS token and reshape the rest
        patch_tokens = x[:, 1:, :]                        # [B, 196, D]
        H = W = int(patch_tokens.size(1) ** 0.5)
        feats = patch_tokens.transpose(1, 2).reshape(B, -1, H, W)  # [B, D, H, W]
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        
        if self.cross_attn is not None and mask is not None:
            feats = self.cross_attn(feats, mask)
        if self.spatial_attn is not None:
            feats = self.spatial_attn(feats, mask)
        if self.attn_pooling is not None:
            pooled = self.attn_pooling(feats, mask)
        else:
            pooled = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)  # [B, D]

        
        return self.classifier(pooled)