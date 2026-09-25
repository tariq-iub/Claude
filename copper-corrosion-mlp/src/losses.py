"""Class-imbalance-aware losses: weighted cross-entropy and focal loss.
Both are compared empirically (src/ablation.py); neither is assumed superior.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(y_train: np.ndarray, n_classes: int, method: str = "inverse-frequency",
                           beta: float = 0.999) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if method == "none":
        w = np.ones(n_classes)
    elif method == "inverse-frequency":
        N = counts.sum()
        w = N / (n_classes * counts)
    elif method == "effective-number":
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"unknown ClassWeightMethod {method}")
    w = w / w.mean()
    return w


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights: np.ndarray | None):
        super().__init__()
        w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weight", w if w is not None else torch.empty(0))
        self._has_weight = class_weights is not None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight if self._has_weight else None
        return F.cross_entropy(logits, targets, weight=w)


class FocalLoss(nn.Module):
    """L_FL = -alpha_t (1-p_t)^gamma log(p_t). alpha_t taken per-class from class_weights."""

    def __init__(self, class_weights: np.ndarray | None, gamma: float = 2.0):
        super().__init__()
        w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("alpha", w if w is not None else torch.empty(0))
        self._has_alpha = class_weights is not None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p_t = log_p.gather(1, targets[:, None]).exp().squeeze(1)
        log_p_t = log_p.gather(1, targets[:, None]).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * log_p_t
        if self._has_alpha:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()
