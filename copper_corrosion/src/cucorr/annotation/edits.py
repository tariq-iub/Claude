"""Pure mask-editing operations used by the annotation app (testable without a UI)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..color_science.color import delta_e, rgb_to_lab


def apply_brush(mask: np.ndarray, painted: np.ndarray, value: int) -> np.ndarray:
    out = mask.copy()
    out[painted.astype(bool)] = value
    return out


def flood_fill(mask: np.ndarray, image: np.ndarray, seed: tuple[int, int], value: int,
               tolerance: float = 8.0) -> np.ndarray:
    """Fill the connected region of pixels within ``tolerance`` dE2000 of the seed colour."""
    lab = rgb_to_lab(image)
    sim = delta_e(lab, lab[seed][None, None, :]) <= tolerance
    lab_cc, _ = ndi.label(sim)
    region = lab_cc == lab_cc[seed]
    return apply_brush(mask, region, value)


def region_fill(mask: np.ndarray, regions: np.ndarray, seed: tuple[int, int], value: int) -> np.ndarray:
    return apply_brush(mask, regions == regions[seed], value)
