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

from .backbones import *


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
        use_spatial_attn: bool = False,
        use_attn_pooling:bool=False
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- backbone
        self.backbone = nnModel(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            remoteclip_ckpt_path=self.hparams.remoteclip_ckpt_path,
            use_attn_mask = use_attn_mask,
            use_spatial_attn = use_spatial_attn,
            use_attn_pooling = use_attn_pooling
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = batch["label"].long()
        logits = self(batch)  # pass whole batch (dict) to forward/backbone
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
        logits = self(batch)  # batch dict in
        return logits.softmax(dim=1)

    # ---- optim ---------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
