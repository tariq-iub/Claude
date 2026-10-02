"""Datasets: synthetic (on the fly or cached) and a manifest-driven real dataset (with optional hardware polarization stack)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import SynthConfig, generate_dataset, make_group, render_view

_TENSOR_KEYS = ("rgb", "lin", "stack", "angles", "S_gt", "normals", "normals_prior", "mask", "pit_mask", "obj_mask", "severity",
                "severity_level", "glare_mask", "rough_gt")


class SyntheticCorrosionDataset(Dataset):
    """groups: iterable of cartridge ids to include; views_per_group views each. Deterministic given (seed, group, view)."""

    def __init__(self, groups: Sequence[int], views_per_group: int = 2, cfg: Optional[SynthConfig] = None, seed: int = 0, cache: bool = True):
        self.cfg, self.seed, self.vpg = cfg or SynthConfig(), seed, views_per_group
        self.index = [(g, v) for g in groups for v in range(views_per_group)]
        self.cache: Dict[int, Dict] = {}
        self.use_cache = cache
        self._groups: Dict[int, Dict] = {}

    def __len__(self):
        return len(self.index)

    def group_ids(self) -> np.ndarray:
        return np.array([g for g, _ in self.index])

    def severity_levels(self) -> np.ndarray:
        return np.array([int(self[i]["severity_level"]) for i in range(len(self))])

    def __getitem__(self, i: int) -> Dict:
        if self.use_cache and i in self.cache:
            return self.cache[i]
        g, v = self.index[i]
        if g not in self._groups:
            self._groups[g] = make_group(self.seed * 100003 + g, self.cfg)
        d = render_view(self._groups[g], self.seed * 100003 + g * 997 + v + 1, self.cfg)
        d["group_id"] = g
        if self.use_cache:
            self.cache[i] = d
        return d


def collate(batch: List[Dict]) -> Dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in _TENSOR_KEYS if k in batch[0]}
    out["group_id"] = torch.tensor([b["group_id"] for b in batch])
    out["meta"] = [b["meta"] for b in batch if "meta" in b]
    return out


class RealManifestDataset(Dataset):
    """Real-data dataset driven by a CSV manifest (see docs/DATASET_PROTOCOL.md).
    Required columns: image_path, mask_path, group_id (cartridge id), session_id.
    Optional: I0_path, I45_path, I90_path, I135_path (+ I22_5_path, ...) hardware analyzer images (linear, same registration),
    severity_level, severity_fraction. Masks are class-index PNGs following vpc.corrosion.segmentation.CLASS_NAMES.
    Image sizes must be divisible by 16 (resize/pad in ``transform``)."""

    ANGLE_COLS = {0.0: "I0_path", 22.5: "I22_5_path", 45.0: "I45_path", 67.5: "I67_5_path", 90.0: "I90_path",
                  112.5: "I112_5_path", 135.0: "I135_path", 157.5: "I157_5_path"}

    def __init__(self, manifest_csv: str, root: str = ".", indices: Optional[np.ndarray] = None, transform=None):
        import pandas as pd
        self.df = pd.read_csv(manifest_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.root, self.transform = root, transform
        self.angles = [a for a, c in self.ANGLE_COLS.items() if c in self.df.columns and self.df[c].notna().all()]

    def __len__(self):
        return len(self.df)

    def _img(self, p):
        from PIL import Image
        return np.asarray(Image.open(os.path.join(self.root, p))).astype(np.float32)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._img(r["image_path"])
        img = img / (65535.0 if img.max() > 255 else 255.0)
        d = {"rgb": torch.from_numpy(img[..., :3]).permute(2, 0, 1), "mask": torch.from_numpy(self._img(r["mask_path"]).astype(np.int64)),
             "group_id": int(r["group_id"])}
        d["lin"] = d["rgb"]  # replaced downstream by srgb_to_linear if images are gamma-encoded
        if self.angles:
            st = []
            for a in self.angles:
                x = self._img(r[self.ANGLE_COLS[a]])
                st.append(torch.from_numpy(x / (65535.0 if x.max() > 255 else 255.0)).permute(2, 0, 1)[:3])
            d["stack"], d["angles"] = torch.stack(st), torch.tensor(self.angles)
        if "severity_fraction" in r:
            d["severity"] = torch.tensor(float(r["severity_fraction"]))
        if "severity_level" in r:
            d["severity_level"] = torch.tensor(int(r["severity_level"]))
        d["pit_mask"] = (d["mask"] == 6).long()
        d["meta"] = {"image_path": r["image_path"]}
        return self.transform(d) if self.transform else d
