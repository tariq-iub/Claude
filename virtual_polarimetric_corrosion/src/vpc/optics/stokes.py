"""Stokes parameters, DoLP, AoLP and circular-safe angular utilities.

Frame convention: S1 > 0 means polarization along the image x-axis (analyzer angle 0),
S2 > 0 along +45 deg (counter-clockwise as seen by the camera). I_theta = 1/2 (S0 + S1 cos2t + S2 sin2t).
Linear polarimetry cannot recover S3; helpers assume S3 = 0 unless stated.
"""
from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


def stokes_from_four(i0: Tensor, i45: Tensor, i90: Tensor, i135: Tensor) -> Tensor:
    """Classical 4-angle linear Stokes: S0 = (sum)/2, S1 = I0 - I90, S2 = I45 - I135. Returns (...,3) stacked on a new dim 0."""
    s0 = 0.5 * (i0 + i45 + i90 + i135)
    return torch.stack([s0, i0 - i90, i45 - i135], dim=0)


def design_matrix(theta: Tensor) -> Tensor:
    """(A, 3) matrix B with I_theta = B @ [S0, S1, S2]."""
    return 0.5 * torch.stack([torch.ones_like(theta), torch.cos(2 * theta), torch.sin(2 * theta)], dim=-1)


def fit_linear_stokes(stack: Tensor, theta: Tensor, dim: int = 0) -> Tensor:
    """Least-squares linear Stokes from analyzer measurements.
    stack: tensor with analyzer axis at ``dim`` (size A >= 3); theta: (A,) radians.
    Returns tensor with Stokes axis (size 3) at ``dim``."""
    theta = theta.to(stack.dtype)
    pinv = torch.linalg.pinv(design_matrix(theta))  # (3, A)
    moved = stack.movedim(dim, 0)
    s = torch.einsum("sa,a...->s...", pinv, moved)
    return s.movedim(0, dim)


def dolp(S: Tensor, dim: int = 0, eps: float = 1e-6, clamp: bool = True) -> Tensor:
    """sqrt(S1^2 + S2^2)/S0 with a safe sqrt (differentiable at 0). ``clamp`` limits to [0,1] (physical bound)."""
    s0, s1, s2 = S.unbind(dim=dim)[:3]
    d = torch.sqrt(s1 ** 2 + s2 ** 2 + eps ** 2) / s0.clamp_min(eps)
    return d.clamp(0.0, 1.0) if clamp else d


def aolp(S: Tensor, dim: int = 0) -> Tensor:
    """AoLP = 1/2 atan2(S2, S1) in (-pi/2, pi/2]. Circular with period pi."""
    s = S.unbind(dim=dim)
    return 0.5 * torch.atan2(s[2], s[1])


def aolp_features(S: Tensor, dim: int = 0, eps: float = 1e-6) -> Tensor:
    """(cos 2phi, sin 2phi) computed without atan2 so they are smooth and defined (0,0) when unpolarized."""
    s = S.unbind(dim=dim)
    norm = torch.sqrt(s[1] ** 2 + s[2] ** 2 + eps ** 2)
    return torch.stack([s[1] / norm, s[2] / norm], dim=dim)


def stokes_from_dolp_aolp(s0: Tensor, rho: Tensor, phi: Tensor, dim: int = 0) -> Tensor:
    return torch.stack([s0, s0 * rho * torch.cos(2 * phi), s0 * rho * torch.sin(2 * phi)], dim=dim)


def wrapped_angular_diff(phi1: Tensor, phi2: Tensor) -> Tensor:
    """Distance between two AoLP values on the circle of period pi: min(|d|, pi-|d|), d=(phi1-phi2) mod pi. Range [0, pi/2]."""
    d = torch.remainder(phi1 - phi2, math.pi)
    return torch.minimum(d, math.pi - d)


def is_physical(S: Tensor, dim: int = 0, tol: float = 1e-6) -> Tensor:
    """Boolean mask: S0 >= 0 and sqrt(S1^2+S2^2) <= S0 (+tol), i.e. realizable partially polarized light."""
    s = S.unbind(dim=dim)
    return (s[0] >= -tol) & (torch.sqrt(s[1] ** 2 + s[2] ** 2) <= s[0] * (1 + tol) + tol)
