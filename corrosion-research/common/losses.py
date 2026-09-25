"""Shared loss functions: CE, Dice, boundary (gradient-consistency) loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import IGNORE_INDEX, NUM_CLASSES


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6):
    num_classes = logits.shape[1]
    valid = (target != ignore_index)
    target_clamped = target.clamp(min=0, max=num_classes - 1)
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs, onehot = probs * valid, onehot * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    """Encourages predicted-probability gradients to align with GT boundary gradients."""
    num_classes = logits.shape[1]
    valid = (target != ignore_index)
    target_clamped = target.clamp(min=0, max=num_classes - 1)
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()

    def grad_mag(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    pg = grad_mag(probs)
    tg = grad_mag(onehot)
    v = valid.unsqueeze(1).float()
    diff = (pg - tg).abs() * v
    return diff.sum() / v.sum().clamp(min=1.0)


class CombinedSegLoss(nn.Module):
    """CE + Dice + boundary, the default multi-term objective used across architectures."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, boundary_weight=0.5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
        self.ce_w, self.dice_w, self.b_w = ce_weight, dice_weight, boundary_weight

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_dice = dice_loss(logits, target)
        l_b = boundary_loss(logits, target)
        total = self.ce_w * l_ce + self.dice_w * l_dice + self.b_w * l_b
        return total, {"ce": l_ce.item(), "dice": l_dice.item(), "boundary": l_b.item()}
