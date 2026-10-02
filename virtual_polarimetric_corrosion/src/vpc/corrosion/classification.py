"""Image-level and region-level classification derived from the segmentation posterior."""
from __future__ import annotations

import torch

from .segmentation import BACKGROUND, CORRODED, HEALTHY

Tensor = torch.Tensor


def corroded_area_fraction(prob: Tensor) -> Tensor:
    """Soft corroded-area fraction of the *object* (non-background) region. prob (B,K,H,W) -> (B,)."""
    obj = 1.0 - prob[:, BACKGROUND]
    corr = prob[:, list(CORRODED)].sum(1)
    return corr.sum((1, 2)) / obj.sum((1, 2)).clamp_min(1.0)


def healthy_vs_corroded(prob: Tensor, area_thresh: float = 0.01) -> Tensor:
    """Task 1: image-level label (1 = corroded) from the soft area fraction; threshold is a config choice
    (tune on validation, never on test)."""
    return (corroded_area_fraction(prob) > area_thresh).long()
