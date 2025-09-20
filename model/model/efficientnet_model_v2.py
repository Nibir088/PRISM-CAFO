# classifier/model/model/efficientnet.py
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.model.components.evidence import EvidenceHead
from model.model.components.pooling import ComponentPooling, GAPConf
from model.model.components.concepts import ConceptHead
from model.model.components.glue import ClassifierGlue
from model.model.cross_attention import CrossAttentionMaskFusion
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling


class EfficientNetBuilder:
    @staticmethod
    def _build_efficientnet(
        name: str,
        weights_enum,
        num_classes: int,
        *,
        use_attn_mask: bool = True,
        use_spatial_attn: bool = True,
        use_attn_pooling: bool = True,
        use_evidence_map: bool = True,
        use_component_pool: bool = False,
        use_conf_gap: bool = False,
        use_concepts: bool = False,
        use_rel_feats: bool = True,
        linear_head: bool = False,
        deepsets_hidden: int = 256,
        deepsets_out: int = 256,
        freeze_backbone: bool = False
    ) -> nn.Module:
        m = getattr(models, name)(weights=weights_enum)

        # Visual dim before the classifier Linear
        d_visual = m.classifier[-1].in_features

        # Keep spatial map: use EfficientNet.features (NOT children()[:-1], which includes avgpool)
        backbone = m.features  # -> [B, D, H', W']

        # Optional modules (parity with other builders)
        cross_attn   = CrossAttentionMaskFusion(d_visual) if use_attn_mask else None
        spatial_attn = SpatialAttention(d_visual, 7) if use_spatial_attn else None
        attn_pooling = MaskAttentionPooling(d_visual, 7) if use_attn_pooling else None
        evidence     = EvidenceHead(d_visual) if use_evidence_map else None
        pool_ds      = ComponentPooling(d_in=d_visual, d_hidden=deepsets_hidden, d_out=d_visual) if use_component_pool else None

        # Classifier input dims depend on pooling path
        d_feat = d_visual
        d_rel  = 12 if use_rel_feats else 0
        head   = ClassifierGlue(d_feat=d_feat, d_rel=d_rel, num_classes=num_classes, linear_head=linear_head)

        concept_barn = concept_pond = concept_feed = None
        if use_concepts:
            concept_barn = ConceptHead(d_visual)
            concept_pond = ConceptHead(d_visual)
            concept_feed = ConceptHead(d_visual)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        return EfficientNetWrapper(
            backbone=backbone,
            d_visual=d_visual,
            head=head,
            evidence=evidence,
            pool_ds=pool_ds,
            cross_attn=cross_attn,
            spatial_attn=spatial_attn,
            attn_pooling=attn_pooling,
            use_conf_gap=use_conf_gap,
            use_concepts=use_concepts,
            use_rel_feats=use_rel_feats,
            concept_barn=concept_barn,
            concept_pond=concept_pond,
            concept_feed=concept_feed,
        )


class EfficientNetWrapper(nn.Module):
    """
    Mirrors Swin/ResNet/ViT/RemoteCLIP wrappers.

    Batch keys:
      - 'rgb': [B,3,H,W]
      - optional 'masks': [B,K,H,W] (K=7)
      - optional 'mask_conf': [B,1,H,W]

    Returns self.last_out dict:
      logits, feat_map, S, gap_feat, R, p_barn, p_pond, p_feed
    """
    def __init__(
        self,
        backbone: nn.Module,
        d_visual: int,
        head: nn.Module,
        evidence: Optional[nn.Module],
        pool_ds: Optional[nn.Module],
        cross_attn: Optional[nn.Module],
        spatial_attn: Optional[nn.Module],
        attn_pooling: Optional[nn.Module],
        *,
        use_conf_gap: bool,
        use_concepts: bool,
        use_rel_feats: bool,
        concept_barn: Optional[nn.Module],
        concept_pond: Optional[nn.Module],
        concept_feed: Optional[nn.Module],
    ):
        super().__init__()
        self.backbone = backbone
        self.d_visual = d_visual

        self.head = head
        self.evidence = evidence
        self.pool_ds = pool_ds

        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling

        self.use_conf_gap = use_conf_gap
        self.use_concepts = use_concepts
        self.use_rel_feats = use_rel_feats

        self.concept_barn = concept_barn
        self.concept_pond = concept_pond
        self.concept_feed = concept_feed
        
        self.gap_conf = GAPConf()

        self.last_out: Dict[str, Any] = {}

    @staticmethod
    def _rel_feats(M: torch.Tensor) -> torch.Tensor:
        # light-weight priors
        from model.model.components.priors import soft_center, soft_area, chamfer_distance
        area_all = soft_area(M)
        out = []
        for i in range(7):
            area_x = soft_area(M[:,i:i+1])
            area_x = 1e-6 + area_x / (area_all+1e-6)
            out.append(area_x.unsqueeze(-1))
        
        dist  = chamfer_distance(M[:,0:1], M[:,1:2])                     # [B,1]
        out.append(dist)
        # area_b= soft_area(M_b).unsqueeze(-1)                   # [B,1]
        # area_p= soft_area(M_p).unsqueeze(-1)                   # [B,1]
        # ratio = area_b / (area_p + area_b + 1e-6)             # [B,1]
        # area_all =  
        return torch.cat(out, dim=-1)  # [B,4]

    def forward(self, batch: Dict[str, Any], return_dict: bool = True):
        x: torch.Tensor = batch["rgb"]
        M7: Optional[torch.Tensor] = batch.get("masks")
        C: Optional[torch.Tensor]  = batch.get("mask_conf")

        # Spatial feature map from EfficientNet
        feat_map: torch.Tensor = self.backbone(x)          # [B, D, H', W']
        B, D, H, W = feat_map.shape

        # Evidence before attentional mods
        S = self.evidence(feat_map) if self.evidence is not None else None

        # Resize masks to feature map
        M7p = None
        if M7 is not None:
            M7p = F.interpolate(M7.float(), size=(H, W), mode="nearest")

        # Cross-attention + spatial gating
        if self.cross_attn is not None and M7p is not None:
            feat_map = self.cross_attn(feat_map, M7p)
        if self.spatial_attn is not None and M7p is not None:
            feat_map = self.spatial_attn(feat_map, M7p)

        # Pool to global vector
        if self.use_conf_gap and C is not None:
            C_resized = F.interpolate(C, size=(H, W), mode="bilinear", align_corners=False)
            gap_feat = self.gap_conf(feat_map, C_resized)       # [B, D]
        elif self.attn_pooling is not None and M7p is not None:
            gap_feat = self.attn_pooling(feat_map, M7p)    # [B, D]
        elif self.pool_ds is not None and M7p is not None:
            _, gap_feat = self.pool_ds(feat_map, M7p)      # [B, deepsets_out]
        else:
            gap_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # [B, D]

        # Optional relation priors (barn vs pond)
        R = None
        if self.use_rel_feats and M7p is not None and M7p.shape[1] >= 2:
            M_b = M7p[:, 0:1]
            M_p = M7p[:, 1:2]
            R = self._rel_feats(M7p)                  # [B,4]
            R = torch.cat([R, batch['prior_prob']], dim=1)  # shape: (B, 8)

        # Optional concept heads on spatial map
        p_barn = p_pond = p_feed = None
        if self.use_concepts:
            p_barn = self.concept_barn(feat_map)
            p_pond = self.concept_pond(feat_map)
            p_feed = self.concept_feed(feat_map)

        # Classifier input
        all_feature = torch.cat([gap_feat, R.to(gap_feat.device)], dim=-1) if R is not None else gap_feat
        logits = self.head(all_feature)

        self.last_out = {
            "logits": logits,
            "feat_map": feat_map,
            "S": S,
            "gap_feat": gap_feat,
            "R": R,
            "p_barn": p_barn,
            "p_pond": p_pond,
            "p_feed": p_feed,
        }
        return self.last_out if return_dict else self.last_out

    
    
    
    
    
# # classifier/model/model/efficientnet.py
# from typing import Dict, Any, Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# from model.model.components.evidence import EvidenceHead
# from model.model.components.pooling import ComponentPooling, gap_conf
# from model.model.components.concepts import ConceptHead
# from model.model.components.glue import ClassifierGlue
# from model.model.cross_attention import CrossAttentionMaskFusion
# from model.model.spatial_attention import SpatialAttention
# from model.model.attn_pooling import MaskAttentionPooling


# class EfficientNetBuilder:
#     @staticmethod
#     def _build_efficientnet(
#         name: str,
#         weights_enum,
#         num_classes: int,
#         *,
#         use_attn_mask: bool = True,
#         use_spatial_attn: bool = True,
#         use_attn_pooling: bool = True,
#         use_evidence_map: bool = True,
#         use_component_pool: bool = False,
#         use_conf_gap: bool = False,
#         use_concepts: bool = False,
#         use_rel_feats: bool = True,
#         linear_head: bool = False,
#         deepsets_hidden: int = 256,
#         deepsets_out: int = 256,
#     ) -> nn.Module:
#         m = getattr(models, name)(weights=weights_enum)

#         # Dim before the final Linear
#         d_visual = m.classifier[-1].in_features

#         # KEEP the previous behavior: children()[:-1] -> includes avgpool, excludes classifier
#         backbone = nn.Sequential(*list(m.children())[:-1])  # [B, D, 1, 1] after avgpool

#         # Optional modules (kept for API parity; operate on 1x1 maps here)
#         cross_attn   = CrossAttentionMaskFusion(d_visual) if use_attn_mask else None
#         spatial_attn = SpatialAttention(d_visual, 7) if use_spatial_attn else None
#         attn_pooling = MaskAttentionPooling(d_visual, 7) if use_attn_pooling else None
#         evidence     = EvidenceHead(d_visual) if use_evidence_map else None
#         pool_ds      = ComponentPooling(d_in=d_visual, d_hidden=deepsets_hidden, d_out=deepsets_out) if use_component_pool else None

#         # Classifier input dims depend on pooling path
#         d_feat = deepsets_out if use_component_pool and attn_pooling is None and not use_conf_gap else d_visual
#         d_rel  = 4 if use_rel_feats else 0
#         head   = ClassifierGlue(d_feat=d_feat, d_rel=d_rel, num_classes=num_classes, linear_head=linear_head)

#         concept_barn = concept_pond = concept_feed = None
#         if use_concepts:
#             concept_barn = ConceptHead(d_visual)
#             concept_pond = ConceptHead(d_visual)
#             concept_feed = ConceptHead(d_visual)

#         return EfficientNetWrapper(
#             backbone=backbone,
#             d_visual=d_visual,
#             head=head,
#             evidence=evidence,
#             pool_ds=pool_ds,
#             cross_attn=cross_attn,
#             spatial_attn=spatial_attn,
#             attn_pooling=attn_pooling,
#             use_conf_gap=use_conf_gap,
#             use_concepts=use_concepts,
#             use_rel_feats=use_rel_feats,
#             concept_barn=concept_barn,
#             concept_pond=concept_pond,
#             concept_feed=concept_feed,
#         )


# class EfficientNetWrapper(nn.Module):
#     """
#     Mirrors Swin/ResNet/ViT/RemoteCLIP wrappers.

#     Batch keys:
#       - 'rgb': [B,3,H,W]
#       - optional 'masks': [B,K,H,W] (K=7)
#       - optional 'mask_conf': [B,1,H,W]

#     Returns self.last_out dict:
#       logits, feat_map, S, gap_feat, R, p_barn, p_pond, p_feed
#     """
#     def __init__(
#         self,
#         backbone: nn.Module,
#         d_visual: int,
#         head: nn.Module,
#         evidence: Optional[nn.Module],
#         pool_ds: Optional[nn.Module],
#         cross_attn: Optional[nn.Module],
#         spatial_attn: Optional[nn.Module],
#         attn_pooling: Optional[nn.Module],
#         *,
#         use_conf_gap: bool,
#         use_concepts: bool,
#         use_rel_feats: bool,
#         concept_barn: Optional[nn.Module],
#         concept_pond: Optional[nn.Module],
#         concept_feed: Optional[nn.Module],
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.d_visual = d_visual

#         self.head = head
#         self.evidence = evidence
#         self.pool_ds = pool_ds

#         self.cross_attn = cross_attn
#         self.spatial_attn = spatial_attn
#         self.attn_pooling = attn_pooling

#         self.use_conf_gap = use_conf_gap
#         self.use_concepts = use_concepts
#         self.use_rel_feats = use_rel_feats

#         self.concept_barn = concept_barn
#         self.concept_pond = concept_pond
#         self.concept_feed = concept_feed

#         self.last_out: Dict[str, Any] = {}

#     @staticmethod
#     def _rel_feats(M_b: torch.Tensor, M_p: torch.Tensor) -> torch.Tensor:
#         from model.model.components.priors import soft_center, soft_area, chamfer_distance
#         cy_b, cx_b = soft_center(M_b); cy_p, cx_p = soft_center(M_p)
#         dist  = chamfer_distance(M_b, M_p)                 # [B,1]
#         area_b= soft_area(M_b).unsqueeze(-1)               # [B,1]
#         area_p= soft_area(M_p).unsqueeze(-1)               # [B,1]
#         ratio = area_b / (area_p + area_b + 1e-6)          # [B,1]
#         return torch.cat([dist, area_b, area_p, ratio], dim=-1)  # [B,4]

#     def forward(self, batch: Dict[str, Any], return_dict: bool = False):
#         x = batch["rgb"]
#         M7 = batch.get("masks")
#         C  = batch.get("mask_conf")

#         # Backbone already includes avgpool -> [B, D, 1, 1]
#         feat_map = self.backbone(x)
#         B, D, H, W = feat_map.shape  # H=W=1

#         # Evidence before any attentional modulation
#         S = self.evidence(feat_map) if self.evidence is not None else None

#         # Resize masks to (1,1) if provided
#         M7p = None
#         if M7 is not None:
#             M7p = F.interpolate(M7.float(), size=(H, W), mode="nearest")

#         # Cross-attn + spatial gating (degenerate at 1x1 but API-consistent)
#         if self.cross_attn is not None and M7p is not None:
#             feat_map = self.cross_attn(feat_map, M7p)
#         if self.spatial_attn is not None and M7p is not None:
#             feat_map = self.spatial_attn(feat_map, M7p)

#         # Pool to global vector (already 1x1)
#         if self.use_conf_gap and C is not None:
#             C_resized = F.interpolate(C, size=(H, W), mode="bilinear", align_corners=False)
#             gap_feat = gap_conf(feat_map, C_resized)       # [B, D]
#         elif self.attn_pooling is not None and M7p is not None:
#             gap_feat = self.attn_pooling(feat_map, M7p)    # [B, D] (acts like a learned gate at 1x1)
#         elif self.pool_ds is not None and M7p is not None:
#             _, gap_feat = self.pool_ds(feat_map, M7p)      # [B, deepsets_out]
#         else:
#             gap_feat = feat_map.flatten(1)                 # [B, D]

#         # Optional relation priors (barn vs pond), computed on 1x1 masks
#         R = None
#         if self.use_rel_feats and M7p is not None and M7p.shape[1] >= 2:
#             M_b = M7p[:, 0:1]
#             M_p = M7p[:, 1:2]
#             R = self._rel_feats(M_b, M_p)                  # [B,4]

#         # Optional concept heads
#         p_barn = p_pond = p_feed = None
#         if self.use_concepts:
#             p_barn = self.concept_barn(feat_map)
#             p_pond = self.concept_pond(feat_map)
#             p_feed = self.concept_feed(feat_map)

#         # Classifier
#         all_feature = torch.cat([gap_feat, R.to(gap_feat.device)], dim=-1) if R is not None else gap_feat
#         logits = self.head(all_feature)

#         self.last_out = {
#             "logits": logits,
#             "feat_map": feat_map,
#             "S": S,
#             "gap_feat": gap_feat,
#             "R": R,
#             "p_barn": p_barn,
#             "p_pond": p_pond,
#             "p_feed": p_feed,
#         }
#         return self.last_out if return_dict else self.last_out
