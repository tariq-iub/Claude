"""Pitting cues and PittingHead. Colour alone is a weak pitting cue; pits are small, locally dark, show a shadow/
highlight dipole under oblique light, discontinuous normals at their rim and elevated multi-scale Laplacian energy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.curvature import normal_discontinuity
from ..inverse.roughness import local_std, multiscale_laplacian_energy

Tensor = torch.Tensor
N_CUES = 4


def pit_cue_maps(lum: Tensor, normals: Tensor | None = None) -> Tensor:
    """(B,4,H,W): [multiscale-Laplacian energy, shadow-highlight dipole, normal discontinuity, local roughness(std)]."""
    lap = multiscale_laplacian_energy(lum, (1, 2, 4))
    mx, mn = F.max_pool2d(lum, 5, 1, 2), -F.max_pool2d(-lum, 5, 1, 2)
    dipole = (mx - mn) * torch.exp(-((lum - mn) / (mx - mn + 1e-4)) * 4)  # strong when centre sits at local minimum of a high-range window
    nd = normal_discontinuity(normals) if normals is not None else torch.zeros_like(lum)
    rough = local_std(lum, 5)
    cues = torch.cat([lap, dipole, nd, rough], 1)
    scale = cues.flatten(2).quantile(0.99, dim=2).view(cues.shape[0], -1, 1, 1).clamp_min(1e-4)
    return (cues / scale).clamp(0, 2)


class PittingHead(nn.Module):
    """Binary pit logit from decoder features concatenated with (pooled) handcrafted cues."""

    def __init__(self, feat_ch: int, n_cues: int = N_CUES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + n_cues, feat_ch, 3, padding=1, groups=1), nn.BatchNorm2d(feat_ch), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 1, 1))

    def forward(self, feat: Tensor, cues: Tensor) -> Tensor:
        cues = F.adaptive_avg_pool2d(cues, feat.shape[-2:]) if cues.shape[-2:] != feat.shape[-2:] else cues
        return self.net(torch.cat([feat, cues], 1))
