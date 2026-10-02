"""Surface-normal utilities and (weak) monocular normal priors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def normalize(n: Tensor, dim: int = 1) -> Tensor:
    return n / n.norm(dim=dim, keepdim=True).clamp_min(1e-6)


def angular_error_deg(n1: Tensor, n2: Tensor, dim: int = 1) -> Tensor:
    c = (normalize(n1, dim) * normalize(n2, dim)).sum(dim).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


def zenith_azimuth_map(n: Tensor) -> tuple[Tensor, Tensor]:
    """n: (B,3,H,W) -> zenith (B,H,W), azimuth (B,H,W)."""
    return torch.acos(n[:, 2].clamp(-1, 1)), torch.atan2(n[:, 1], n[:, 0])


def shading_gradient_normals(lum: Tensor, strength: float = 2.0) -> Tensor:
    """WEAK shape-from-shading style prior: treat smoothed luminance gradients as slope cues, n ~ (-g*gx, -g*gy, 1).
    Ambiguous (bas-relief/convex-concave, albedo-shading confound); intended only as an input prior to learning."""
    k = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=lum.dtype, device=lum.device) / 8.0
    lp = F.pad(lum, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(lp, k.view(1, 1, 3, 3))
    gy = F.conv2d(lp, k.t().contiguous().view(1, 1, 3, 3))
    n = torch.cat([-strength * gx, strength * gy, torch.ones_like(gx)], 1)
    return normalize(n)


class NormalHead(nn.Module):
    """1x1 conv head predicting unit normals from features; z forced non-negative (camera-facing)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        n = self.conv(x)
        n = torch.cat([n[:, :2], F.softplus(n[:, 2:3]) + 1e-3], 1)
        return normalize(n)
