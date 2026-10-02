"""Measured-polarization input features for the colour-only pipeline.

Physics (Stokes fit, DoLP, AoLP) is reused from ``vpc.optics.stokes`` (virtual_polarimetric_corrosion); this module only
adds feature selection and Stokes-aware augmentation. Frame convention follows vpc: S1 > 0 along image x (analyzer 0 deg),
S2 > 0 along +45 deg.

Augmentation must act on the Stokes vector, not on the feature maps of a stack: rotating the scene by k*90 deg rotates the
polarization angle by k*90 deg, i.e. (s1, s2) -> (-1)^k (s1, s2); a horizontal flip mirrors the angle, s2 -> -s2.
DoLP and S0 are invariant to both.
"""
from __future__ import annotations

import os
import sys

import torch

try:
    from vpc.optics import stokes as _stokes
except ImportError:  # sibling package in this repo; avoids requiring `pip install -e`
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "virtual_polarimetric_corrosion", "src"))
    from vpc.optics import stokes as _stokes

# mode -> number of extra input channels
POL_MODES = {"none": 0, "dolp": 1, "stokes": 3, "stokes_aolp": 5, "stack": None}


def pol_channels(mode: str, n_angles: int = 4) -> int:
    if mode not in POL_MODES:
        raise ValueError(f"unknown pol_mode {mode!r}; choose from {sorted(POL_MODES)}")
    return n_angles if mode == "stack" else POL_MODES[mode]


def stokes_from_stack(stack: torch.Tensor, angles_deg, colour: int = 1) -> torch.Tensor:
    """stack: (A, 3, H, W) linear analyzer images -> (3, H, W) Stokes (S0, S1, S2) of one colour channel (default green)."""
    theta = torch.as_tensor(angles_deg, dtype=stack.dtype) * torch.pi / 180.0
    return _stokes.fit_linear_stokes(stack[:, colour], theta, dim=0)


def augment_stokes(S: torch.Tensor, k_rot90: int = 0, hflip: bool = False) -> torch.Tensor:
    """Apply a spatial rot90 (k times, counter-clockwise) and/or horizontal flip to a (3,H,W) Stokes map with the matching
    polarization transform. Flip is applied first, then rotation."""
    if hflip:
        S = torch.flip(S, dims=[-1])
        S = torch.stack([S[0], S[1], -S[2]])
    if k_rot90 % 4:
        S = torch.rot90(S, k_rot90 % 4, dims=(-2, -1))
        if k_rot90 % 2:
            S = torch.stack([S[0], -S[1], -S[2]])
    return S


def pol_features(S: torch.Tensor, mode: str) -> torch.Tensor:
    """(3,H,W) Stokes -> (P,H,W) features. 'stack' is handled by the caller (raw intensities)."""
    if mode == "none":
        return S.new_zeros((0,) + S.shape[-2:])
    s0 = S[0].clamp_min(1e-6)
    s1n, s2n = S[1] / s0, S[2] / s0
    rho = _stokes.dolp(S, dim=0)
    if mode == "dolp":
        return rho.unsqueeze(0)
    if mode == "stokes":
        return torch.stack([s1n, s2n, rho])
    if mode == "stokes_aolp":
        c, s = _stokes.aolp_features(S, dim=0)
        return torch.stack([s1n, s2n, rho, c, s])
    raise ValueError(f"unsupported pol_mode {mode!r} here")


class PolFusionWrapper(torch.nn.Module):
    """Model-agnostic late fusion: wraps any ``model(rgb)->logits`` and adds a residual logit correction computed from
    [polarization features, softmax(logits)]. Input is ``x = [rgb(3) | pol(P)]``. With P = 0 it is exactly the base model,
    so the colour-only arm and the polarization arms share one code path. Early fusion inside individual architectures is a
    later step; this wrapper is the cheap first test of whether the extra channels carry class information."""

    def __init__(self, base: torch.nn.Module, n_classes: int, p_channels: int, hidden: int = 8):
        super().__init__()
        self.base, self.p = base, p_channels
        if p_channels:
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(p_channels + n_classes, hidden, 3, padding=1), torch.nn.GELU(),
                torch.nn.Conv2d(hidden, n_classes, 1))
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        out = self.base(x[:, :3])
        logits = out[0] if isinstance(out, tuple) else out
        if not self.p:
            return out
        corr = self.head(torch.cat([x[:, 3:], torch.softmax(logits.detach(), 1)], 1))
        return logits + corr
