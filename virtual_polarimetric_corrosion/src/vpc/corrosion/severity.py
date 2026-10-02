"""Corroded-area percentage and severity (ordinal). Severity thresholds are PLACEHOLDERS to be fixed by a domain
standard / expert panel before any data is collected; they are config, not a scientific result."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification import corroded_area_fraction

Tensor = torch.Tensor

DEFAULT_BINS = (0.01, 0.05, 0.15, 0.30)  # area-fraction cut-points -> 5 ordinal levels 0..4


def area_to_level(frac: Tensor, bins=DEFAULT_BINS) -> Tensor:
    return torch.bucketize(frac, torch.tensor(bins, dtype=frac.dtype, device=frac.device))


def ordinal_targets(level: Tensor, n_levels: int) -> Tensor:
    """(B,) int -> (B, n_levels-1) binary [level > t]."""
    return (level.unsqueeze(1) > torch.arange(n_levels - 1, device=level.device).unsqueeze(0)).float()


def ordinal_loss(logits: Tensor, level: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, ordinal_targets(level, logits.shape[1] + 1))


def decode_ordinal(logits: Tensor) -> Tensor:
    return (torch.sigmoid(logits) > 0.5).long().sum(1)


class SeverityHead(nn.Module):
    """Global descriptor [GAP(features), soft corroded-area, mean pit prob] -> severity score in [0,1] + ordinal logits."""

    def __init__(self, feat_ch: int, n_levels: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feat_ch + 2, 32), nn.ReLU(inplace=True), nn.Linear(32, 1 + (n_levels - 1)))

    def forward(self, feat: Tensor, seg_prob: Tensor, pit_prob: Tensor) -> dict:
        area = corroded_area_fraction(seg_prob).unsqueeze(1)
        g = torch.cat([feat.mean((2, 3)), area, pit_prob.mean((1, 2, 3)).unsqueeze(1)], 1)
        o = self.mlp(g)
        return {"severity": torch.sigmoid(o[:, 0]), "severity_ordinal": o[:, 1:], "area_frac": area.squeeze(1)}
