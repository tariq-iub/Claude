"""Curvature and normal-discontinuity cues from a normal field (used by the pitting module)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _sobel(x: Tensor):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    c = x.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = kx.transpose(-1, -2)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(xp, kx, groups=c), F.conv2d(xp, ky, groups=c)


def mean_curvature_proxy(n: Tensor) -> Tensor:
    """Divergence of the in-plane normal field / n_z, a mean-curvature proxy (B,1,H,W), sign: convex > 0 (image y down)."""
    gx, _ = _sobel(n[:, 0:1])
    _, gy = _sobel(n[:, 1:2])
    return -(gx + gy) / n[:, 2:3].clamp_min(0.2)


def normal_discontinuity(n: Tensor) -> Tensor:
    """Max angular jump (radians) between a pixel's normal and its 4-neighbours: (B,1,H,W). High at pit rims/edges."""
    out = torch.zeros_like(n[:, :1])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = torch.roll(n, shifts=(dy, dx), dims=(2, 3))
        c = (n * m).sum(1, keepdim=True).clamp(-1, 1)
        out = torch.maximum(out, torch.acos(c))
    return out
