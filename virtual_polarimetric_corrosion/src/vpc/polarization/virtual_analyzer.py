"""Differentiable virtual analyzer: I_theta = 1/2 (S0 + S1 cos2t + S2 sin2t).

Stokes layout used by the network code: (B, C, 3, H, W) (colour channel C, Stokes axis 3). Output (B, A, C, H, W).
The *physical meaning* of S_hat is "estimated"; see docs/LIMITATIONS.md.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from ..optics.mueller import linear_polarizer_mueller

Tensor = torch.Tensor

CANONICAL_ANGLES_DEG = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
FOUR_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


def deg2rad(x):
    return torch.as_tensor(x, dtype=torch.float32) * (math.pi / 180.0)


def dense_angles_deg(step: float = 5.0) -> list[float]:
    n = int(round(180.0 / step))
    return [i * step for i in range(n)]


class VirtualAnalyzer(nn.Module):
    """VirtualAnalyzer(theta): theta in [0, 180) degrees (any real value accepted; period 180 deg).

    mode='closed_form' uses the Malus-compatible expression; mode='mueller' multiplies by M_P(theta)
    (reference path; supports finite extinction ratio via ``t_min``). Both coincide for an ideal polarizer
    (verified in tests). ``learnable_extinction`` exposes t_min as a parameter to calibrate against hardware.
    """

    def __init__(self, mode: str = "closed_form", t_max: float = 1.0, t_min: float = 0.0,
                 learnable_extinction: bool = False, stokes_dim: int = 2):
        super().__init__()
        assert mode in ("closed_form", "mueller")
        self.mode, self.t_max, self.stokes_dim = mode, t_max, stokes_dim
        if learnable_extinction:
            self.log_tmin = nn.Parameter(torch.tensor(math.log(max(t_min, 1e-4))))
        else:
            self.register_buffer("log_tmin", torch.tensor(math.log(max(t_min, 1e-12))), persistent=False)
            self._fixed_tmin = t_min
        self.learnable = learnable_extinction

    @property
    def t_min(self) -> Tensor:
        return torch.exp(self.log_tmin) if self.learnable else torch.tensor(self._fixed_tmin)

    def forward(self, S: Tensor, theta_deg) -> Tensor:
        theta = deg2rad(theta_deg).to(S.device, S.dtype).reshape(-1)
        d = self.stokes_dim
        s0, s1, s2 = S.select(d, 0), S.select(d, 1), S.select(d, 2)  # (B,C,H,W)
        c, s = torch.cos(2 * theta), torch.sin(2 * theta)
        shape = (1, -1) + (1,) * (s0.dim() - 1)  # (1, A, 1, 1, 1)
        c, s = c.view(shape), s.view(shape)
        s0, s1, s2 = s0.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(1)
        if self.mode == "closed_form" and not self.learnable and float(self.t_min) == 0.0 and self.t_max == 1.0:
            return 0.5 * (s0 + s1 * c + s2 * s)
        tmax, tmin = self.t_max, self.t_min.to(S)
        a, b = 0.5 * (tmax + tmin), 0.5 * (tmax - tmin)
        return a * s0 + b * (s1 * c + s2 * s)  # first row of the diattenuator Mueller matrix (S3 = 0)

    def reference_mueller(self, S4: Tensor, theta_deg) -> Tensor:
        """Reference implementation: full 4x4 Mueller product on (…,4) Stokes vectors."""
        theta = deg2rad(theta_deg).to(S4)
        M = linear_polarizer_mueller(theta, self.t_max, float(self.t_min)).to(S4)
        return torch.einsum("...ij,...j->...i", M, S4)[..., 0]


def virtual_stack(S: Tensor, angles_deg: Sequence[float] = CANONICAL_ANGLES_DEG, analyzer: VirtualAnalyzer | None = None) -> Tensor:
    """Counterfactual polarization stack (B, A, C, H, W)."""
    analyzer = analyzer or VirtualAnalyzer()
    return analyzer(S, angles_deg)
