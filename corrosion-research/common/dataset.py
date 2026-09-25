"""
Synthetic corrosion-image dataset generator + a real-folder dataset loader.

No public, license-clear pixel-level copper-corrosion dataset ships with this
repository, so `SyntheticCorrosionDataset` procedurally renders plausible
training/validation data (metallic base + oxide blobs with class-consistent
Lab statistics, illumination gradients, shadows, specular noise and dirt) so
every architecture in this repo is trainable and testable end-to-end out of
the box. Swap in `FolderCorrosionDataset` once real annotated images/masks
are available (expects `images/*.png` and `masks/*.png` with integer class
ids, same basenames).

Classes
-------
0 healthy / uncorroded (metallic copper)
1 Cu2O   / cuprite      (red-brown)
2 CuO    / tenorite     (black / dark brown)
3 CuCl   / nantokite    (pale grey-green, powdery)
4 Cu2Cl(OH)3 / atacamite-type (bright green)
5 ignore / uncertain
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 6
CLASS_NAMES = ["healthy", "cuprite", "tenorite", "nantokite", "atacamite", "ignore"]
IGNORE_INDEX = 5

# Approximate mean sRGB colour per class used to seed synthetic rendering.
_CLASS_RGB = {
    0: (0.72, 0.45, 0.20),   # bright copper
    1: (0.55, 0.22, 0.12),   # cuprite red-brown
    2: (0.12, 0.09, 0.08),   # tenorite near-black
    3: (0.75, 0.78, 0.72),   # nantokite pale grey
    4: (0.20, 0.55, 0.30),   # atacamite green
}


def _smoothstep_blob(h, w, cx, cy, rx, ry, rng: np.random.RandomState):
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    # irregular boundary via low-frequency noise perturbation
    noise = rng.normal(0, 0.15, size=(4, 4)).repeat(h // 4 + 1, axis=0).repeat(w // 4 + 1, axis=1)[:h, :w]
    d = d + noise
    mask = d < 1.0
    return mask


@dataclass
class SyntheticCorrosionDataset(Dataset):
    length: int = 200
    height: int = 128
    width: int = 128
    seed: int = 0
    illumination_aug: bool = True

    def __post_init__(self):
        self._rng_base = self.seed

    def __len__(self):
        return self.length

    def _render(self, rng: np.random.RandomState):
        h, w = self.height, self.width
        base = np.array(_CLASS_RGB[0], dtype=np.float32)
        img = np.tile(base, (h, w, 1)) + rng.normal(0, 0.02, size=(h, w, 3)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.int64)

        n_blobs = rng.randint(2, 6)
        for _ in range(n_blobs):
            cls = rng.randint(1, 5)
            cx, cy = rng.randint(0, w), rng.randint(0, h)
            rx, ry = rng.randint(w // 10, w // 3), rng.randint(h // 10, h // 3)
            blob = _smoothstep_blob(h, w, cx, cy, rx, ry, rng)
            colour = np.array(_CLASS_RGB[cls], dtype=np.float32)
            colour = colour + rng.normal(0, 0.03, size=3).astype(np.float32)
            img[blob] = colour
            mask[blob] = cls

        # a thin "uncertain" ring around blob boundaries (simulate annotation
        # ambiguity at oxide transition zones) mapped to the ignore class
        from scipy.ndimage import binary_dilation, binary_erosion

        boundary = np.zeros_like(mask, dtype=bool)
        for c in range(1, 5):
            m = mask == c
            if m.any():
                ring = binary_dilation(m, iterations=1) & ~binary_erosion(m, iterations=1)
                boundary |= ring
        ignore_mask = boundary & (rng.rand(h, w) < 0.15)
        mask[ignore_mask] = IGNORE_INDEX

        # illumination gradient + shadow + specular/dirt noise
        if self.illumination_aug:
            gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / max(h, w)
            grad = 1.0 + 0.25 * (gx * xx + gy * yy)
            img *= grad[..., None]
            if rng.rand() < 0.5:
                sx, sy, sr = rng.randint(0, w), rng.randint(0, h), rng.randint(w // 4, w // 2)
                yy2, xx2 = np.mgrid[0:h, 0:w]
                shadow = np.exp(-(((xx2 - sx) ** 2 + (yy2 - sy) ** 2) / (2 * sr ** 2)))
                img *= (1.0 - 0.35 * shadow)[..., None]
            dirt = rng.normal(0, 0.015, size=(h, w, 3))
            img += dirt

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return img, mask

    def __getitem__(self, idx):
        rng = np.random.RandomState(self._rng_base * 100003 + idx)
        img, mask = self._render(rng)
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t


class FolderCorrosionDataset(Dataset):
    """Loads real data once available: images/*.png + masks/*.png (same stem)."""

    def __init__(self, root: str, transform=None):
        self.images = sorted(glob.glob(os.path.join(root, "images", "*.png")))
        self.masks_dir = os.path.join(root, "masks")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.images[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.masks_dir, stem + ".png")
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path), dtype=np.int64)
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask).long()
        if self.transform:
            img_t, mask_t = self.transform(img_t, mask_t)
        return img_t, mask_t


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
