# classifier/model/classifier_module.py
from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAveragePrecision,
)
from model.model.components.priors import *
from .backbones_v3 import *


class CAFOClassifier(pl.LightningModule):
    """
    Thin Lightning wrapper around a pluggable backbone that returns class logits.
    Expects batches shaped like your DataModule's collate_simple:
      batch["rgb"]   -> [B,3,H,W]
      batch["label"] -> [B]
    """

    def __init__(
        self,
        num_classes: int = 6,
        lr: float = 1e-4,
        model_name: str = "resnet18",
        remoteclip_ckpt_path: Optional[str] = None,
        freeze_backbone: bool = False,
        channels_last: bool = False,
        weight_decay: float = 0.0,
        use_bf16_metrics: bool = True,  # kept for compatibility
        use_attn_mask: bool = False,
        use_evidence_map: bool = False,
        use_component_pool: bool = False,
        use_conf_gap: bool = True,
        use_concepts: bool = False,
        use_rel_feats: bool = True,
        linear_head: bool = False,
        deepsets_hidden: int = 256,
        deepsets_out: int = 256,
        use_spatial_attn: bool = True, use_attn_pooling:bool=True
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- backbone
        self.backbone = nnModel(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            remoteclip_ckpt_path=self.hparams.remoteclip_ckpt_path,
            use_attn_mask = use_attn_mask,
            use_evidence_map = use_evidence_map,
            use_component_pool = use_component_pool,
            use_conf_gap = use_conf_gap,
            use_concepts = use_concepts,
            use_rel_feats = use_rel_feats,
            linear_head = linear_head,
            deepsets_hidden  = deepsets_hidden,
            deepsets_out = deepsets_out,
            use_spatial_attn = use_spatial_attn, use_attn_pooling=use_attn_pooling,
            freeze_backbone = freeze_backbone
        )

        # if freeze_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False

        # ---- metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_map   = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

    # ---- forward / steps -----------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Pass the full batch dict to the backbone.
        Optionally switch only the image tensor to channels_last.
        """
        if self.hparams.channels_last and "rgb" in batch:
            # Do not mutate caller's dict in-place: create a shallow copy
            rgb = batch["rgb"].to(memory_format=torch.channels_last)
            batch = {**batch, "rgb": rgb}
        return self.backbone(batch)  # backbone is responsible for reading needed keys
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self(batch)  # forward pass through backbone
        y = batch["label"].long()                                # [B]
        logits = out['logits']                                   # [B, num_classes]
        S = out['S']                                             # [B, 1, H', W'] or similar attention map
        mask = batch['masks'].float()                            # [B, 7, H, W]
        

        # --- Classification loss
        l_cls = F.cross_entropy(logits, y)

        # --- Auxiliary interpretability & sparsity losses
        # l_align = L_align(S, M_U)
        # l_focus = L_focus(S, M_U)
        # l_cir = L_circ(mask[:, 1:2])            # Assuming [B, 7, H, W] → M_U[:, 1] is a channel/component
        # l_l1 = L_sparse(S)
        # l_tv = tv_loss(S)

        # --- Sufficiency/Necessity Losses
#         with torch.no_grad():
#             def f_logits(inp):
#                 return self.backbone({**batch, "rgb": inp})['logits']

#         l_suf, l_nec = suff_nec_losses(f_logits, batch["rgb"], batch['mask_conf'], y)

        # --- Weighted loss combination
        # loss = (
        #     0.7 * l_cls +
        #     0.2 * l_suf +
        #     0.1 * l_nec 
        # )
        
        loss = (
            1 * l_cls 
            
        )

        return loss, logits

        
    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = batch["label"].long()
        # logits = self(batch)  # pass whole batch (dict) to forward/backbone
        # class_weights = torch.tensor([0.3690, 2.1253, 6.7056, 0.7039, 4.0076], device=logits.device)
        loss,logits = self.compute_losses(batch)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)
        return {"loss": loss, "logits": logits, "probs": probs, "y": y, "preds": preds}

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.train_acc(out["preds"], out["y"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        self.val_acc(out["preds"], out["y"])
        self.val_f1(out["preds"], out["y"])
        self.val_map(out["probs"], out["y"])
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_acc",  self.val_acc, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_f1",   self.val_f1,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)
        self.log("val_map",  self.val_map, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)['logits']  # batch dict in
        return logits.softmax(dim=1)

    # ---- optim ---------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
