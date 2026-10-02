"""Roughness proxies from a single image. Roughness is NOT identifiable from RGB alone; these are weak cues
(highlight spread, local micro-contrast) used as inputs/priors to the learned estimator."""
from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _rpad(x: Tensor, p: int) -> Tensor:
    """Replicate padding: zero padding would create false edges / variance along the image border."""
    return F.pad(x, (p, p, p, p), mode="replicate")


def local_std(x: Tensor, k: int = 7) -> Tensor:
    xp = _rpad(x, k // 2)
    m = F.avg_pool2d(xp, k, 1)
    return torch.sqrt((F.avg_pool2d(xp * xp, k, 1) - m * m).clamp_min(1e-10))


def multiscale_laplacian_energy(lum: Tensor, scales=(1, 2, 4)) -> Tensor:
    """Mean absolute DoG/Laplacian response over scales (B,1,H,W): micro-topography/texture cue."""
    outs = []
    kern = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
    for s in scales:
        x = F.avg_pool2d(lum, s, s) if s > 1 else lum
        r = F.conv2d(_rpad(x, 1), kern).abs()
        outs.append(F.interpolate(r, size=lum.shape[-2:], mode="bilinear", align_corners=False))
    return torch.stack(outs).mean(0)


def roughness_proxy(lum: Tensor, spec_prob: Tensor) -> Tensor:
    """Bounded [0,1] heuristic: rough surfaces give broad, low-contrast highlights and high micro-contrast.
    r_proxy = sigmoid(a * (texture_energy_norm) - b * highlight_peakiness). Constants are arbitrary priors (documented)."""
    tex = multiscale_laplacian_energy(lum)
    tex = tex / (tex.flatten(1).quantile(0.95, dim=1).view(-1, 1, 1, 1) + 1e-6)
    peak = local_std(spec_prob, 9)
    peak = peak / (peak.flatten(1).max(1).values.view(-1, 1, 1, 1) + 1e-6)
    return torch.sigmoid(3.0 * (tex - 0.5) - 2.0 * (peak - 0.3)).clamp(0.05, 1.0)
