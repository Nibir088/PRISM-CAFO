from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.model.components.evidence import EvidenceHead
from model.model.components.pooling import ComponentPooling, GAPConf
from model.model.components.concepts import ConceptHead
from model.model.components.glue import ClassifierGlue
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling
try:
    from model.model.cross_attention import CrossAttentionMaskFusion
except Exception:
    CrossAttentionMaskFusion = None

class SwinBuilder:
    @staticmethod
    def _build_swin_b(
        num_classes: int,
        pretrained: bool = True,
        use_attn_mask: bool = True,
        use_evidence_map: bool = True,
        use_component_pool: bool = False,
        use_conf_gap: bool = False,
        use_concepts: bool = False,
        use_rel_feats: bool = True,
        linear_head: bool = False,
        deepsets_hidden: int = 256,
        deepsets_out: int = 256,
        use_spatial_attn: bool = True, use_attn_pooling:bool=True,
        freeze_backbone: bool = False
    ) -> nn.Module:
        weights = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        swin = models.swin_b(weights=weights)
        d_visual = swin.head.in_features
        evidence = EvidenceHead(d_visual)
        swin.head = nn.Identity()
        
        if use_attn_mask:
            cross_attn = CrossAttentionMaskFusion(d_visual)
        else:
            cross_attn = None
        if use_spatial_attn:
            spatial_attn = SpatialAttention(d_visual, 7)
        else:
            spatial_attn = None
        if use_attn_pooling:
            attn_pooling = MaskAttentionPooling(d_visual, 7)
        else:
            attn_pooling = None
        if freeze_backbone:
            for p in swin.parameters():
                p.requires_grad = False
        # classifier = nn.Linear(visual_dim, num_classes)
        return SwinWrapper(
            swin=swin, d_visual=d_visual, num_classes=num_classes,
            cross_attn=cross_attn,
            use_evidence_map=use_evidence_map,
            use_component_pool=use_component_pool,
            use_conf_gap=use_conf_gap,
            use_concepts=use_concepts,
            use_rel_feats=use_rel_feats,
            linear_head=linear_head,
            deepsets_hidden=deepsets_hidden,
            deepsets_out=deepsets_out,
            evidence = evidence,
            spatial_attn = spatial_attn,
            attn_pooling = attn_pooling
        )

class SwinWrapper(nn.Module):
    """
    Forward returns logits (API-stable). Extras cached in self.last_out.
    Expects batch keys: 'rgb', optional 'masks', 'mask_conf'.
    """
    def __init__(
        self, swin: nn.Module, d_visual: int, num_classes: int,
        cross_attn: Optional[nn.Module],
        use_evidence_map: bool, use_component_pool: bool,
        use_conf_gap: bool, use_concepts: bool,
        use_rel_feats: bool, linear_head: bool,
        deepsets_hidden: int, deepsets_out: int,
        evidence, spatial_attn, attn_pooling
    ):
        super().__init__()
        self.swin = swin
        self.cross_attn = cross_attn
        self.spatial_attn = spatial_attn
        self.attn_pooling = attn_pooling
        # self.visual_dim = swin.head.in_features

        self.use_evidence_map  = use_evidence_map
        self.use_component_pool= use_component_pool
        self.use_conf_gap      = use_conf_gap
        self.use_concepts      = use_concepts
        self.use_rel_feats     = use_rel_feats
        
        self.gap_conf = GAPConf()

        self.evidence = evidence if use_evidence_map else None
        

        if use_concepts:
            self.concept_barn = ConceptHead(d_visual)
            self.concept_pond = ConceptHead(d_visual)
            self.concept_feed = ConceptHead(d_visual)

        d_feat = d_visual  # global or conf-GAP produce [B,D]
        
        self.pool     = ComponentPooling(d_in=d_visual, d_hidden=deepsets_hidden, d_out=d_visual) if use_component_pool else None
        
        d_agg  = deepsets_out if use_component_pool else 0
        d_rel  = 12 if use_rel_feats else 0
        self.head = ClassifierGlue(d_feat=d_feat, d_rel=d_rel, num_classes=num_classes, linear_head=linear_head)

        self.last_out: Dict[str, Any] = {}

        # Hook fallback if .features not present
        self._feat_map = None
        def _hook(_, __, out):
            self._feat_map = out
            return out
        if not hasattr(self.swin, "features"):
            for m in reversed(list(self.swin.modules())):
                if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Conv2d):
                    m.register_forward_hook(_hook); break

    def _get_feat_map(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.swin, "features"):
            return self.swin.features(x)
        _ = self.swin(x)
        if self._feat_map is None:
            raise RuntimeError("Cannot capture Swin feature map; upgrade torchvision or keep features(x).")
        return self._feat_map

    def _rel_feats(self, M: torch.Tensor) -> torch.Tensor:
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
    def forward(self, batch: Dict[str, Any], return_dict: bool = False):
        x  = batch["rgb"]                                 # [B,3,H,W]
        M7 = batch.get("masks", None)                     # [B,K,H,W] or None
        C  = batch.get("mask_conf", None)                 # [B,1,H,W] or None
        # print(M7,C)
        feat_map = self._get_feat_map(x)                  # [B,D,H',W']
        feat_map = feat_map.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        B, D, Hp, Wp = feat_map.shape

        S = self.evidence(feat_map) if self.evidence is not None else None
        
        feats_global = self.swin(x)                       # [B,D]
        
        M7p = F.interpolate(M7.float(), size=(Hp, Wp), mode="nearest")  # [B, K, Hp, Wp]
        if self.cross_attn is not None and M7 is not None:
            
            feat_map = self.cross_attn(feat_map, M7p)  # apply attention at feature map level
            # print(feat_map.shape)
        if self.spatial_attn is not None:
            feat_map = self.spatial_attn(feat_map, M7p)
        
        
        
        if self.use_conf_gap:
            C_resized = F.interpolate(C, size=(Hp, Wp), mode="bilinear", align_corners=False)
            gap_feat = self.gap_conf(feat_map, C_resized)  # [B, D]
        elif self.attn_pooling is not None:
            gap_feat = self.attn_pooling(feat_map, M7p)
        elif self.pool is not None:
            M7p = F.interpolate(M7.float(), size=(Hp,Wp), mode="nearest").clamp(0,1)
            _, gap_feat = self.pool(feat_map, M7p)             # [B,deepsets_out]
        else:
            gap_feat = F.adaptive_avg_pool2d(feat_map, 1).squeeze(-1).squeeze(-1)  # [B, D]

        
        
        R = None#torch.zeros((B,0), device=x.device)
        if self.use_rel_feats and M7 is not None and M7.shape[1] >= 2:
            M7p = F.interpolate(M7.float(), size=(Hp,Wp), mode="nearest")
            
            R = self._rel_feats(M7p)                 # [B,4]
            R = torch.cat([R, batch['prior_prob']], dim=1)  # shape: (B, 8)
            # print(R)
        # print('R: ',R.shape)
        p_barn = p_pond = p_feed = None
        if self.use_concepts:
            p_barn = self.concept_barn(feat_map)
            p_pond = self.concept_pond(feat_map)
            p_feed = self.concept_feed(feat_map)
        
        # print('Feat Map after cross: ',feat_map.shape)
        all_feature = gap_feat
        if R!=None:
            all_feature = torch.cat([all_feature, R.to(x.device)], dim=-1)
        logits = self.head(
            all_feature
        )
        # print(self.head)

        self.last_out = {
            "logits": logits, "feat_map": feat_map, "S": S,
            "gap_feat": gap_feat, "R": R,
            "p_barn": p_barn, "p_pond": p_pond, "p_feed": p_feed,
        }
        return self.last_out #self.last_out #self.last_out if return_dict else logits
