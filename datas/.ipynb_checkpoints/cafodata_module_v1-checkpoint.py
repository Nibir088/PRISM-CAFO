# cafodata_module.py
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
import lightning as pl  # pip install lightning
from .cafodataset import CAFODataset, _COMPONENT_TYPES

# super-simple collate (expects all keys present and same shapes)
def collate_simple(batch):
    rgb   = torch.stack([b["rgb"] for b in batch], dim=0)          # [B,3,H,W]
    b4    = torch.stack([b["b4"]  for b in batch], dim=0)          # [B,1,H,W]
    label = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # masks/conf assumed same shapes across batch
    masks = torch.stack([b["masks"]     for b in batch], dim=0)    # [B,T,H,W]
    conf  = torch.stack([b["mask_conf"] for b in batch], dim=0)    # [B,H,W]

    # carry non-tensor metadata
    types    = batch[0]["mask_label"]   # fixed order component names
    category = [b["category"] for b in batch]
    state    = [b["state"]    for b in batch]

    return {
        "rgb": rgb,
        "b4": b4,
        "label": label,
        "masks": masks,          # already float in [0,1]; no clamp needed
        "mask_conf": conf,       # [0,1]
        "mask_label": types,
        "category": category,
        "state": state,
    }



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
        shuffle_train: bool = True,
        val_ratio: float = 0.15,           # used only if val_jsonl is None
        split_seed: int = 42,
        drop_last: bool = False,
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
        self.shuffle_train = shuffle_train
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.drop_last = drop_last

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_simple,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_simple,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_simple,
            drop_last=False,
        )
