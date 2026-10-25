"""Pixel feature extraction shared by approaches B, C and F."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from ..color_science.color import rgb_to_hsv, rgb_to_lab


def pixel_features(image: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """Return (H, W, F) float32 features. Groups: rgb, lab, hsv, chroma, intensity,
    texture (local std of L* at two scales + gradient magnitude), spatial (normalised y, x)."""
    h, w = image.shape[:2]
    feats = []
    lab = rgb_to_lab(image)
    if "rgb" in groups:
        feats.append(image.astype(np.float32) / 255.0)
    if "lab" in groups:
        feats.append(lab / np.array([100.0, 128.0, 128.0]))
    if "hsv" in groups:
        hsv = rgb_to_hsv(image)
        ang = 2 * np.pi * hsv[..., 0]
        feats.append(np.stack([np.cos(ang), np.sin(ang), hsv[..., 1], hsv[..., 2]], -1))
    if "chroma" in groups:
        feats.append((np.hypot(lab[..., 1], lab[..., 2]) / 128.0)[..., None])
    if "intensity" in groups:
        feats.append((image.astype(np.float32).mean(-1) / 255.0)[..., None])
    if "texture" in groups:
        L = lab[..., 0] / 100.0
        t = []
        for s in (2, 5):
            m = ndi.uniform_filter(L, 2 * s + 1)
            t.append(np.sqrt(np.maximum(ndi.uniform_filter(L * L, 2 * s + 1) - m * m, 0)))
        t.append(ndi.gaussian_gradient_magnitude(L, 1.5))
        feats.append(np.stack(t, -1))
    if "spatial" in groups:
        yy, xx = np.mgrid[0:h, 0:w]
        feats.append(np.stack([yy / max(h - 1, 1), xx / max(w - 1, 1)], -1))
    if not feats:
        raise ValueError("no feature groups selected")
    return np.concatenate(feats, -1).astype(np.float32)

