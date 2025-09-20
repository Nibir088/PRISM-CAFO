# cafodata_module.py
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from functools import partial

# Use a package-safe import (this file lives next to cafodataset.py)
from .cafodataset import CAFODataset

# ---------- Collate functions -------------------------------------------------

def collate_cls(batch: List[Dict[str, Any]], *, channels_last: bool = False) -> Dict[str, torch.Tensor]:
    """Minimal, fast collate for classification: only 'rgb' and 'label'."""
    rgb = torch.stack([b["rgb"] for b in batch], 0)  # [B,3,H,W]
    if channels_last:
        rgb = rgb.contiguous(memory_format=torch.channels_last)

    label = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)
    return {"rgb": rgb, "label": label}


def collate_full(batch: List[Dict[str, Any]], *, channels_last: bool = False) -> Dict[str, Any]:
    """
    Full but safe collate:
    - Always stacks 'rgb' and 'label'
    - Only stacks optional tensors if present (b4, masks, mask_conf)
    - Carries lightweight metadata without heavy stacking
    """
    out: Dict[str, Any] = {}

    rgb = torch.stack([b["rgb"] for b in batch], 0)
    if channels_last:
        rgb = rgb.contiguous(memory_format=torch.channels_last)
    out["rgb"] = rgb

    out["label"] = torch.as_tensor([b["label"] for b in batch], dtype=torch.long)

    first = batch[0]

    # Optional tensors
    if "b4" in first and first["b4"] is not None:
        out["b4"] = torch.stack([b["b4"] for b in batch], 0)
    if "masks" in first and first["masks"] is not None:
        out["masks"] = torch.stack([b["masks"] for b in batch], 0)
    if "mask_conf" in first and first["mask_conf"] is not None:
        out["mask_conf"] = torch.stack([b["mask_conf"] for b in batch], 0)
    if 'prior_prob' in first and first['prior_prob'] is not None:
        out['prior_prob'] = torch.stack([b['prior_prob'] for b in batch], 0)

    # Lightweight metadata (do not expand to tensors)
    if "mask_label" in first:
        out["mask_label"] = first["mask_label"]  # same order for all samples
    if "category" in first:
        out["category"] = [b.get("category") for b in batch]
    if "state" in first:
        out["state"] = [b.get("state") for b in batch]
    out['path'] = [b.get('path') for b in batch]

    return out


# ---------- DataModule --------------------------------------------------------

class CAFODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_jsonl: str,
        test_jsonl: str,
        val_jsonl: Optional[str] = None,
        *,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
        transform=None,                    # (rgb, b4) -> (rgb, b4); if None uses dataset default
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        shuffle_train: bool = True,
        val_ratio: float = 0.15,           # used only if val_jsonl is None
        split_seed: int = 42,
        drop_last: bool = False,
        # new knobs:
        collate_train_mode: str = "cls",   # "cls" (fast) or "full"
        collate_eval_mode: str = "cls",    # "cls" (fast) or "full"
        channels_last: bool = False,       # speed-up for conv nets
    ):
        super().__init__()
        self.train_jsonl = train_jsonl
        self.test_jsonl  = test_jsonl
        self.val_jsonl   = val_jsonl

        self.resize = resize
        self.transform = transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_train = shuffle_train
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.drop_last = drop_last

        self.channels_last = channels_last
        self.collate_train_mode = collate_train_mode.lower()
        self.collate_eval_mode = collate_eval_mode.lower()

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

        # pick collate functions
        self._collate_train = partial(
            collate_cls if self.collate_train_mode == "cls" else collate_full,
            channels_last=self.channels_last,
        )
        self._collate_eval = partial(
            collate_cls if self.collate_eval_mode == "cls" else collate_full,
            channels_last=self.channels_last,
        )

    def setup(self, stage: Optional[str] = None):
        # Always (re)build datasets on setup
        train_full = CAFODataset(self.train_jsonl, transform=self.transform, resize=self.resize)
        self.test_ds = CAFODataset(self.test_jsonl,  transform=self.transform, resize=self.resize)

        if self.val_jsonl:
            self.val_ds = CAFODataset(self.val_jsonl, transform=self.transform, resize=self.resize)
            self.train_ds = train_full
        else:
            # split off val from train_full
            n = len(train_full)
            if n == 0:
                raise RuntimeError("Train dataset is empty.")
            n_val = max(1, int(round(self.val_ratio * n)))
            n_train = n - n_val
            g = torch.Generator().manual_seed(self.split_seed)
            self.train_ds, self.val_ds = random_split(train_full, [n_train, n_val], generator=g)

    def _loader_kwargs(self) -> Dict[str, Any]:
        """Common DataLoader kwargs with safe handling when num_workers == 0."""
        kw = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        if self.num_workers > 0:
            kw["persistent_workers"] = self.persistent_workers
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def train_dataloader(self) -> DataLoader:
        kw = self._loader_kwargs()
        return DataLoader(
            self.train_ds,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last,
            collate_fn=self._collate_train,
            **kw,
        )

    def val_dataloader(self) -> DataLoader:
        kw = self._loader_kwargs()
        return DataLoader(
            self.val_ds,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_eval,
            **kw,
        )

    def test_dataloader(self) -> DataLoader:
        kw = self._loader_kwargs()
        return DataLoader(
            self.test_ds,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate_eval,
            **kw,
        )
