"""Segmentation classes and losses. Class names are *visually annotated corrosion classes* (image-derived), not
chemically verified compounds."""
from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

CLASS_NAMES = ("background", "healthy_metal", "discoloration", "oxidation_dark", "patina_green", "chloride_deposit", "pitting")
NUM_CLASSES = len(CLASS_NAMES)
CORRODED = (2, 3, 4, 5, 6)   # classes counted as corroded area
HEALTHY, BACKGROUND, PITTING = 1, 0, 6


def dice_loss(logits: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    p = torch.softmax(logits, 1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    d = 1 - (2 * inter + eps) / (den + eps)
    return d[present].mean() if present.any() else d.mean() * 0


def focal_ce(logits: Tensor, target: Tensor, gamma: float = 1.5, weight: Tensor | None = None) -> Tensor:
    logp = F.log_softmax(logits, 1)
    ce = F.nll_loss(logp, target, weight=weight, reduction="none")
    pt = torch.exp(-F.nll_loss(logp, target, reduction="none"))
    return ((1 - pt) ** gamma * ce).mean()


def seg_loss(logits: Tensor, target: Tensor, class_weights: Tensor | None = None, dice_w: float = 1.0, gamma: float = 1.0) -> Tensor:
    return focal_ce(logits, target, gamma, class_weights) + dice_w * dice_loss(logits, target)


def sobel_mag(x: Tensor) -> Tensor:
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 4.0
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx)
    gy = F.conv2d(xp, kx.transpose(-1, -2))
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def boundary_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Edge-preservation: BCE between Sobel boundaries of the predicted soft label map and the target label map
    (computed on the one-hot channels, summed over classes)."""
    p = torch.softmax(logits, 1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    B, K, H, W = p.shape
    ep = sobel_mag(p.reshape(B * K, 1, H, W)).clamp(0, 1)
    et = sobel_mag(oh.reshape(B * K, 1, H, W)).clamp(0, 1)
    return F.binary_cross_entropy(ep, et)
