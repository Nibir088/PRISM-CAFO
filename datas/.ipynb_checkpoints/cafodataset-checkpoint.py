# cafodataset.py
import os, json
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import rasterio
_COMPONENT_TYPES = [
    "barn",
    "manure_pond",
    "silage_storage",
    "silo",
    "feedlot",
    "silage_bunker",
    "building"
]

class CAFODataset(Dataset):
    """
    Yields:
      {
        "rgb":  FloatTensor [3,H,W],
        "b4":   FloatTensor [1,H,W],
        "label": int in {0..4} (Beef=0, Dairy=1, Swine=2, Poultry=3, other=4),
        "category": str,
        "state": str or None,
        "masks": BoolTensor [K,H,W] or None
      }
    """
    def __init__(
        self,
        jsonl_path: str,
        transform: Optional[Callable] = None,       # (rgb, b4) -> (rgb, b4)
        resize: Optional[Union[int, Tuple[int,int]]] = None,  # int or (H,W)
    ):
        self.rows = self._read_jsonl(jsonl_path)
        self.resize = (resize, resize) if isinstance(resize, int) else resize
        self.prior_prob = pd.read_csv('/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/CAFOSat/classifier/datas/dataset/farms_agcensus.csv')
        
        self.prior_prob = self.prior_prob.dropna(subset=['beef', 'poultry', 'swine', 'dairy'])

        # 5-class mapping
        self.label_map = {"Beef":4, "Dairy":1, "Swine":2, "Poultry":3,"Negative":0}
        # default transform if none: scale to [0,1] by dtype
        if transform is None:
            def _default(rgb: torch.Tensor, b4: torch.Tensor):
                # assume uint8 -> /255, uint16 -> /65535, else as-is
                denom = 255.0
                rgb = rgb.to(torch.float32) / (denom if denom>1 else 1.0)
                b4  = b4.to(torch.float32)  / (denom if denom>1 else 1.0)
                return rgb, b4
            self.transform = _default
        else:
            self.transform = transform

    def __len__(self): return len(self.rows)

    # ---- helpers ----
    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        with open(path, "r") as f:
            return [json.loads(l) for l in f if l.strip()]

    
    @staticmethod
    def _load_4bands(path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            if src.count >= 4:
                arr = src.read([1,2,3,4])           # (4,H,W)
            else:
                arr = src.read()                     # (C,H,W)
                if src.count < 4:
                    pad = np.zeros((4 - src.count, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
        return arr
    

    @staticmethod
    def _read_npz_masks(
        path: str,
        types=_COMPONENT_TYPES,
        default_hw=(833, 833),
    ):
        """
        Returns:
          masks_t: torch.uint8 [T,H,W]  (0/1) per component type in 'types'
          conf_t:  torch.float32 [1,H,W] confidence heatmap in [0,1]
        """
        H0, W0 = default_hw

        if not os.path.exists(path):
            masks_zero = torch.zeros((len(types), H0, W0), dtype=torch.uint8)
            conf_zero  = torch.zeros((1, H0, W0), dtype=torch.float32)
            return masks_zero, conf_zero, types

        d = np.load(path, allow_pickle=True)

        # --- masks ---
        m = d["masks"] if "masks" in d.files else d[d.files[0]]          # (K,H,W) or (H,W)
        if m.ndim == 2:
            m = m[None, ...]                                             # -> (K=1,H,W)
        m_bool = (m > 0)

        # --- labels (optional) ---
        if "mask_labels" in d.files:
            labels = d["mask_labels"]
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            labels = [str(x).lower() for x in labels]
        else:
            labels = None

        H, W = m_bool.shape[-2], m_bool.shape[-1]

        # --- per-type binary masks (0/1 uint8) ---
        if labels is None:
            masks_np = np.stack([np.zeros((H, W), dtype=np.uint8) for _ in types], axis=0)
        else:
            out = []
            for t in types:
                idxs = [i for i, lab in enumerate(labels) if lab == t]
                if len(idxs) == 0:
                    out.append(np.zeros((H, W), dtype=np.uint8))
                else:
                    out.append(np.any(m_bool[idxs, ...], axis=0).astype(np.uint8))
            masks_np = np.stack(out, axis=0)                              # (T,H,W)

        # --- confidence map (always return [1,H,W]) ---
        K = int(m_bool.shape[0])
        if "keep_scores" in d.files and (K > 0):
            ks = np.asarray(d["keep_scores"], dtype=np.float32).reshape(-1)  # (K,)
            if ks.shape[0] == K:
                conf_hw = (m_bool.astype(np.float32) * ks[:, None, None]).max(axis=0)  # (H,W)
                conf_hw = np.clip(conf_hw, 0.0, 1.0)
            else:
                conf_hw = np.zeros((H, W), dtype=np.float32)
        else:
            conf_hw = np.zeros(( H, W), dtype=np.float32)

        conf_np = conf_hw[None, ...]                                     # -> (1,H,W)

        return torch.from_numpy(masks_np), torch.from_numpy(conf_np), types


    def _cat_to_label(self, cat: str) -> int:
        return self.label_map.get(cat, 0)

    # ---- main ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        img_path = r.get("patch_file") or r.get("patch_path")
        
        arr4 = self._load_4bands(img_path)          # (4,H,W), typically uint16
        rgb = torch.from_numpy(arr4[:3])            # [3,H,W]
        b4  = torch.from_numpy(arr4[3:1+3])         # [1,H,W]

        masks, conf, types = self._read_npz_masks(r.get('masks'))  # [K,H,W] or None

        if self.resize is not None:
            size = self.resize  # (H,W)
            rgb = F.interpolate(rgb.unsqueeze(0).float(), size=size, mode="bilinear", align_corners=False).squeeze(0)
            b4  = F.interpolate(b4.unsqueeze(0).float(),  size=size, mode="bilinear", align_corners=False).squeeze(0)
            if masks is not None:
                masks = (
                    F.interpolate(masks.unsqueeze(0).float(), size=size, mode="nearest")
                     .squeeze(0)
                     .clamp_(0.0, 1.0)
                )
                conf  = F.interpolate(conf.unsqueeze(0).float(),  size=size, mode="bilinear", align_corners=False).squeeze(0)

        rgb, b4 = self.transform(rgb, b4)  # ONLY apply to rgb+b4
        
        row = self.prior_prob[self.prior_prob['patch_file']==img_path]
        prior_vector = torch.tensor(row[['beef', 'poultry', 'swine', 'dairy']].iloc[0].values, dtype=torch.float32) if not row.empty else torch.full((4,), 0.25)


        return {
            "rgb": rgb,                           # [3,H,W] float
            "b4": b4,                             # [1,H,W] float
            "label": self._cat_to_label(r.get("category", "")),
            "category": r.get("category"),
            "state": r.get("state"),
            "masks": masks,                       # [K,H,W] bool or None
            'mask_conf': conf,
            'mask_label': types,
            'path':img_path,
            'prior_prob':prior_vector
            # 'barn_pond_dist': 
        }
