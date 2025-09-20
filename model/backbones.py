# classifier/model/backbones.py
from __future__ import annotations
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
from model.model.resnet import ResNetBuilder
from model.model.efficientnet_model import EfficientNetBuilder
from model.model.dinov2 import DinoV2Builder
from model.model.remoteclip import RemoteCLIPBuilder
from model.model.clip import CLIPBuilder
from model.model.swin_model import SwinBuilder
from model.model.vit_model import ViTBuilder
# Optional open_clip import (better error message if missing)
try:
    import open_clip
    _HAVE_OPEN_CLIP = True
except Exception as _e:
    _HAVE_OPEN_CLIP = False
    _OPEN_CLIP_ERR = _e


class nnModel(nn.Module):
    """
    Unified classifier wrapper. Choose a backbone by name and get a normal nn.Module
    with forward(x) -> logits. Also exposes helpers for transforms/metadata.
    """
    # ---- Norms ----
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        *,
        remoteclip_ckpt_path: Optional[str] = None,
        pretrained: bool = True,
        use_attn_mask: bool = False,
        use_spatial_attn: bool = True,
        use_attn_pooling:bool=True
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        # Build chosen backbone
        # if self.model_name == "resnet18":
        #     self.net = ResNetBuilder._build_resnet("resnet18",
        #                                    models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
        #                                    num_classes,
        #                                    use_attn_mask=use_mask)
        # elif self.model_name == "resnet50":
        #     self.net = ResNetBuilder._build_resnet("resnet50",
        #                                   models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        #                                   num_classes)
        # elif self.model_name == "vit_b_16":
        #     self.net = self._build_vit_b_16(num_classes, pretrained)
        # elif self.model_name == "swin_b":
        #     self.net = self._build_swin_b(num_classes, pretrained)
        # elif self.model_name == "efficientnet_b0":
        #     self.net = self._build_efficientnet("efficientnet_b0",
        #                                         models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None,
        #                                         num_classes)
        # elif self.model_name == "efficientnet_b3":
        #     self.net = self._build_efficientnet("efficientnet_b3",
        #                                         models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None,
        #                                         num_classes)
        # elif self.model_name == "dinov2_vit_b":
        #     self.net = self._build_dinov2_vit_b(num_classes, pretrained)
        # elif self.model_name == "convnext_tiny":
        #     self.net = self._build_convnext_tiny(num_classes, pretrained)
        # elif self.model_name == "clip":
        #     self.net = self._build_clip_vit_b32(num_classes, pretrained=("openai" if pretrained else None))
        # elif self.model_name == "remoteclip":
        #     self.net = self._build_remoteclip_vit_b32(num_classes, remoteclip_ckpt_path)
        if self.model_name == "resnet18":
            self.net = ResNetBuilder._build_resnet(
                "resnet18",
                models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
                num_classes,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "resnet50":
            self.net = ResNetBuilder._build_resnet(
                "resnet50",
                models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                num_classes,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "vit_b_16":
            self.net = ViTBuilder._build_vit_b_16(
                num_classes=num_classes,
                pretrained=pretrained,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "swin_b":
            self.net = SwinBuilder._build_swin_b(
                num_classes=num_classes,
                pretrained=pretrained,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "efficientnet_b0":
            self.net = EfficientNetBuilder._build_efficientnet(
                name="efficientnet_b0",
                weights_enum=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None,
                num_classes=num_classes,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "efficientnet_b3":
            self.net = EfficientNetBuilder._build_efficientnet(
                name="efficientnet_b3",
                weights_enum=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None,
                num_classes=num_classes,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "dinov2_vit_b":
            self.net = DinoV2Builder._build_dinov2_vit_b(
                num_classes=num_classes,
                pretrained=pretrained,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "clip":
            # When pretrained=True, default to "openai"; else None.
            clip_tag = "openai" if pretrained else None
            self.net = CLIPBuilder._build_clip_vit_b32(
                num_classes=num_classes,
                pretrained=clip_tag,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )

        elif self.model_name == "remoteclip":
            self.net = RemoteCLIPBuilder._build_remoteclip_vit_b32(
                num_classes=num_classes,
                ckpt_path=remoteclip_ckpt_path,
                use_attn_mask=use_attn_mask,
                use_spatial_attn= use_spatial_attn,
                use_attn_pooling=use_attn_pooling
            )
        elif self.model_name == "convnext_tiny":
            # (Optional) If you want a ConvNeXt builder, add it similarly to others.
            # For now, keep the simple inline head replace.
            w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.convnext_tiny(weights=w)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
            # Wrap to accept batch dict (rgb only), to match interface:
            self.net = _SimpleBatchWrapper(m)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    # --------- builders ----------
    # @staticmethod
    # def _build_resnet(name: str, weights_enum, num_classes: int) -> nn.Module:
    #     m = getattr(models, name)(weights=weights_enum)
    #     m.fc = nn.Linear(m.fc.in_features, num_classes)
    #     return m

#     @staticmethod
#     def _build_efficientnet(name: str, weights_enum, num_classes: int) -> nn.Module:
#         m = getattr(models, name)(weights=weights_enum)
#         # EfficientNet: classifier = Sequential(Dropout, Linear)
#         m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
#         return m

#     @staticmethod
#     def _build_swin_b(num_classes: int, pretrained: bool) -> nn.Module:
#         w = models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
#         m = models.swin_b(weights=w)
#         m.head = nn.Linear(m.head.in_features, num_classes)
#         return m

#     @staticmethod
#     def _build_convnext_tiny(num_classes: int, pretrained: bool) -> nn.Module:
#         w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
#         m = models.convnext_tiny(weights=w)
#         m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
#         return m

#     @staticmethod
#     def _replace_vit_head(vit: nn.Module, num_classes: int) -> nn.Module:
#         if isinstance(vit.heads, nn.Sequential):
#             in_features = vit.heads[-1].in_features
#         else:
#             in_features = vit.heads.in_features
#         vit.heads = nn.Linear(in_features, num_classes)
#         return vit

#     @classmethod
#     def _build_vit_b_16(cls, num_classes: int, pretrained: bool) -> nn.Module:
#         w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
#         vit = models.vit_b_16(weights=w)
#         return cls._replace_vit_head(vit, num_classes)

#     @staticmethod
#     def _build_dinov2_vit_b(num_classes: int, pretrained: bool) -> nn.Module:
#         # timm "dino" variant (pretrained=True gives self-supervised weights)
#         m = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained)
#         if hasattr(m, "head") and isinstance(m.head, nn.Identity):
#             m.head = nn.Linear(m.num_features, num_classes)
#         else:
#             # Some timm versions expose head as Linear already; just replace
#             in_feats = getattr(m, "num_features", None) or getattr(m.head, "in_features")
#             m.head = nn.Linear(in_feats, num_classes)
#         return m

#     @staticmethod
#     def _build_clip_vit_b32(num_classes: int, pretrained: Optional[str]) -> nn.Module:
#         if not _HAVE_OPEN_CLIP:
#             raise ImportError(
#                 "open_clip is required for CLIP/RemoteCLIP. "
#                 "Install with `pip install open-clip-torch`.\n"
#                 f"Original error: {_OPEN_CLIP_ERR}"
#             )
#         # pretrained can be "openai", "laion2b_s34b_b79k", etc. or None
#         model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=pretrained or None)
#         visual = model.visual
#         return nn.Sequential(visual, nn.Linear(visual.output_dim, num_classes))

#     @staticmethod
#     def _build_remoteclip_vit_b32(num_classes: int, ckpt_path: Optional[str]) -> nn.Module:
#         if not _HAVE_OPEN_CLIP:
#             raise ImportError(
#                 "open_clip is required for CLIP/RemoteCLIP. "
#                 "Install with `pip install open-clip-torch`.\n"
#                 f"Original error: {_OPEN_CLIP_ERR}"
#             )
#         if ckpt_path is None or not os.path.exists(ckpt_path):
#             raise FileNotFoundError("RemoteCLIP checkpoint not found. Provide `remoteclip_ckpt_path`.")
#         model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         model.load_state_dict(ckpt)
#         visual = model.visual
#         return nn.Sequential(visual, nn.Linear(visual.output_dim, num_classes))

    # --------- nn.Module API ----------
    def forward(self, x):
        """
        x: [B,3,H,W] preprocessed to the expected normalization/size by your DataModule.
        returns logits: [B, num_classes]
        """
        # print(x.shape)
        # x = x['rgb']
        x = self.net(x)
        # print(x.shape)
        return x

    # --------- helpers ----------
    @classmethod
    def get_model_metadata(cls, model_name: str) -> Dict[str, object]:
        name = model_name.lower()
        if name in {"clip", "remoteclip"}:
            return {"input_size": 224, "mean": cls._CLIP_MEAN, "std": cls._CLIP_STD, "expects_clip_norm": True}
        return {"input_size": 224, "mean": cls._IMAGENET_MEAN, "std": cls._IMAGENET_STD, "expects_clip_norm": False}

    @classmethod
    def build_default_transform(cls, model_name: str, size: Optional[int] = None):
        meta = cls.get_model_metadata(model_name)
        side = int(size or meta["input_size"])
        return transforms.Compose([
            transforms.Resize(side, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(side),
            transforms.ToTensor(),
            transforms.Normalize(mean=meta["mean"], std=meta["std"]),
        ])
# -------------------------------
# Small utility so inline models still accept "batch" dicts
# -------------------------------
class _SimpleBatchWrapper(nn.Module):
    """
    Wrap a torchvision model that expects tensor input, so it can accept the dict batch format:
      forward({"rgb": x}) -> logits
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, batch):
        x = batch["rgb"]
        return self.net(x)