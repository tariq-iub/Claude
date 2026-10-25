"""Rendering of colour masks, overlays, boundaries and uncertainty maps."""
from __future__ import annotations

import numpy as np

from ..schema import PALETTE, UNCERTAIN


def color_mask(mask: np.ndarray, hidden: set[int] | None = None) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    for cid, rgb in PALETTE.items():
        if hidden and cid in hidden:
            continue
        out[mask == cid] = rgb
    return out


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45, hidden: set[int] | None = None) -> np.ndarray:
    cm = color_mask(mask, hidden).astype(np.float32)
    show = np.ones(mask.shape, bool)
    if hidden:
        show &= ~np.isin(mask, list(hidden))
    out = image.astype(np.float32).copy()
    out[show] = (1 - alpha) * out[show] + alpha * cm[show]
    return np.clip(out, 0, 255).astype(np.uint8)


def boundaries(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32)
    b = np.zeros(m.shape, bool)
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    b[1:, :] |= m[1:, :] != m[:-1, :]
    return b


def boundary_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    b = boundaries(mask)
    cm = color_mask(mask)
    out[b] = cm[b]
    out[b & (mask == 2)] = (255, 255, 0)  # CuO is black in the palette; keep its edge visible
    return out


def uncertainty_map(u: np.ndarray) -> np.ndarray:
    """Map [0,1] uncertainty to an 8-bit grey PNG (0 = certain, 255 = maximally uncertain)."""
    return np.clip(np.round(np.nan_to_num(u) * 255), 0, 255).astype(np.uint8)


def legend_rows() -> list[tuple[int, str, tuple[int, int, int]]]:
    from ..schema import CLASS_NAMES
    rows = [(i, n, PALETTE[i]) for i, n in enumerate(CLASS_NAMES)]
    rows.append((UNCERTAIN, "Unknown/Uncertain", PALETTE[UNCERTAIN]))
    return rows
