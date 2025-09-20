# classifier/model/model/clip.py
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model.components.evidence import EvidenceHead
from model.model.components.pooling import ComponentPooling, GAPConf
from model.model.components.concepts import ConceptHead
from model.model.components.glue import ClassifierGlue
from model.model.cross_attention import CrossAttentionMaskFusion
from model.model.spatial_attention import SpatialAttention
from model.model.attn_pooling import MaskAttentionPooling

try:
    import open_clip
    _HAVE_OPEN_CLIP = True
except Exception as _e:
    _HAVE_OPEN_CLIP = False
    _OPEN_CLIP_ERR = _e


class CLIPBuilder:
    @staticmethod
    def _build_clip_vit_b32(
        num_classes: int,
        pretrained: str = "openai",
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
        if not _HAVE_OPEN_CLIP:
            raise ImportError(
                "open_clip is required for CLIP. Install with `pip install open-clip-torch`.\n"
                f"Original error: {_OPEN_CLIP_ERR}"
            )

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=pretrained)
        visual = model.visual
        if freeze_backbone:
            for p in visual.parameters():
                p.requires_grad = False
        # Token/feature width (ViT-B/32 = 768)
        d_visual = 768

        cross_attn   = CrossAttentionMaskFusion(d_visual) if use_attn_mask else None
        spatial_attn = SpatialAttention(d_visual, 7) if use_spatial_attn else None
        attn_pooling = MaskAttentionPooling(d_visual, 7) if use_attn_pooling else None
        evidence     = EvidenceHead(d_visual) if use_evidence_map else None
        pool_ds      = ComponentPooling(d_in=d_visual, d_hidden=deepsets_hidden, d_out=d_visual) if use_component_pool else None

        d_feat = d_visual
        d_rel  = 12 if use_rel_feats else 0
        head   = ClassifierGlue(d_feat=d_feat, d_rel=d_rel, num_classes=num_classes, linear_head=linear_head)

        concept_barn = concept_pond = concept_feed = None
        if use_concepts:
            concept_barn = ConceptHead(d_visual)
            concept_pond = ConceptHead(d_visual)
            concept_feed = ConceptHead(d_visual)

        return CLIPWrapper(
            visual=visual,
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


class CLIPWrapper(nn.Module):
    """
    Batch keys:
      'rgb': [B,3,H,W], optional 'masks': [B,K,H,W], 'mask_conf': [B,1,H,W]
    Returns self.last_out with: logits, feat_map, S, gap_feat, R, p_barn, p_pond, p_feed
    """
    def __init__(
        self,
        visual: nn.Module,
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
        self.visual = visual
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

    # ---- Token-grid extraction (OpenCLIP ViT) ----
    def _extract_spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        v = self.visual
        x = v.conv1(x)                                     # [B,C,H',W']
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,N,C]
        x = torch.cat([v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)  # [B,N+1,C]
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                             # [N+1,B,C]
        x = v.transformer(x)                               # [N+1,B,C]
        x = x.permute(1, 0, 2)                             # [B,N+1,C]
        return x

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

        tokens = self._extract_spatial_tokens(x)           # [B,N+1,D]
        patch_tokens = tokens[:, 1:, :]                    # drop CLS
        B, N, D = patch_tokens.shape
        Ht = Wt = int(N ** 0.5)
        feat_map = patch_tokens.transpose(1, 2).reshape(B, D, Ht, Wt)

        # Evidence before attentional mods
        S = self.evidence(feat_map) if self.evidence is not None else None

        # Masks to token grid
        M7p = None
        if M7 is not None:
            M7p = F.interpolate(M7.float(), size=(Ht, Wt), mode="nearest")

        # Cross-attention & spatial gating
        if self.cross_attn is not None and M7p is not None:
            feat_map = self.cross_attn(feat_map, M7p)
        if self.spatial_attn is not None and M7p is not None:
            feat_map = self.spatial_attn(feat_map, M7p)

        # Pooling to a global vector
        if self.use_conf_gap and C is not None:
            C_resized = F.interpolate(C, size=(Ht, Wt), mode="bilinear", align_corners=False)
            gap_feat = self.gap_conf(feat_map, C_resized)       # [B, D]
        elif self.attn_pooling is not None and M7p is not None:
            gap_feat = self.attn_pooling(feat_map, M7p)    # [B, D]
        elif self.pool_ds is not None and M7p is not None:
            _, gap_feat = self.pool_ds(feat_map, M7p)      # [B, deepsets_out]
        else:
            gap_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # [B, D]

        # Optional relational priors (barn vs pond)
        R = None
        if self.use_rel_feats and M7p is not None and M7p.shape[1] >= 2:
            M_b = M7p[:, 0:1]
            M_p = M7p[:, 1:2]
            R = self._rel_feats(M7p)                  # [B,4]
            R = torch.cat([R, batch['prior_prob']], dim=1)  # shape: (B, 8)

        # Optional concept heads
        p_barn = p_pond = p_feed = None
        if self.use_concepts:
            p_barn = self.concept_barn(feat_map)
            p_pond = self.concept_pond(feat_map)
            p_feed = self.concept_feed(feat_map)

        # Classifier
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
