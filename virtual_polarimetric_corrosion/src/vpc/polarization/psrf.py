"""Polarimetric Surface Response Field (PSRF).

For each pixel the analyzer-dependent response is approximated by a circular-harmonic series
    R_p(theta) = a0 + a1 cos2t + b1 sin2t + a2 cos4t + b2 sin4t.
First order == linear polarization: a0 = S0/2, a1 = S1/2, b1 = S2/2 (so DoLP = sqrt(a1^2+b1^2)/a0, AoLP = atan2(b1,a1)/2).
Second-order terms are NOT produced by the ideal-analyzer + linear-Stokes physics (a pure linear polarizer yields only
harmonics 0 and 2theta); they can appear with real optics (finite extinction, sensor/optics cross-talk, rotation-dependent
transmission) and are kept only if held-out-angle error / BIC justifies them (see select_order).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

Tensor = torch.Tensor


def harmonic_basis(theta: Tensor, order: int) -> Tensor:
    """(A, 2*order+1) basis [1, cos2t, sin2t, cos4t, sin4t, ...] for angles in radians."""
    cols = [torch.ones_like(theta)]
    for m in range(1, order + 1):
        cols += [torch.cos(2 * m * theta), torch.sin(2 * m * theta)]
    return torch.stack(cols, dim=-1)


class PSRF(nn.Module):
    """Least-squares harmonic fit across the analyzer axis (dim=1 of a (B,A,...) stack)."""

    def __init__(self, angles_deg, order: int = 1):
        super().__init__()
        theta = torch.as_tensor(angles_deg, dtype=torch.float32) * (math.pi / 180.0)
        k = 2 * order + 1
        distinct = torch.unique((torch.remainder(theta, math.pi) * 1e4).round()).numel()
        if distinct < k:
            raise ValueError(f"PSRF order {order} needs >= {k} distinct analyzer angles mod 180 deg; got {distinct}")
        B = harmonic_basis(theta, order)
        # Aliasing check: identifiability requires full column rank
        if torch.linalg.matrix_rank(B) < k:
            raise ValueError(f"Analyzer angles do not identify a {order}-order PSRF (rank-deficient basis, aliasing).")
        self.order = order
        self.register_buffer("theta", theta)
        self.register_buffer("pinv", torch.linalg.pinv(B))  # (k, A)

    def fit(self, stack: Tensor) -> Tensor:
        """stack (B,A,...) -> coefficients (B,k,...)."""
        return torch.einsum("ka,ba...->bk...", self.pinv.to(stack), stack)

    def reconstruct(self, coef: Tensor, theta_deg) -> Tensor:
        th = torch.as_tensor(theta_deg, dtype=coef.dtype, device=coef.device).reshape(-1) * (math.pi / 180.0)
        B = harmonic_basis(th, self.order)
        return torch.einsum("ak,bk...->ba...", B, coef)

    def features(self, coef: Tensor, eps: float = 1e-6) -> dict:
        """Named per-pixel features (each (B,1,...) or (B,...)): a0,a1,b1,(a2,b2), DoLP_hat, cos2phi, sin2phi, higher-order energy."""
        a0, a1, b1 = coef[:, 0], coef[:, 1], coef[:, 2]
        mag = torch.sqrt(a1 ** 2 + b1 ** 2 + eps ** 2)
        out = {
            "a0": a0, "a1": a1, "b1": b1,
            "dolp_hat": (mag / a0.clamp_min(eps)).clamp(0, 1),
            "cos2phi": a1 / mag, "sin2phi": b1 / mag,
        }
        if self.order >= 2:
            out["a2"], out["b2"] = coef[:, 3], coef[:, 4]
            out["ho_energy"] = torch.sqrt(coef[:, 3] ** 2 + coef[:, 4] ** 2 + eps ** 2) / a0.clamp_min(eps)
        return out


def select_order(stack: Tensor, angles_deg, max_order: int = 2, holdout_every: int = 2) -> dict:
    """Pick the PSRF order by held-out-angle RMSE (+ BIC) on a dense stack (A >= 2*max_order+1+holdout).
    Returns dict(order -> {'holdout_rmse', 'bic'}) and 'best'. Intended for hardware-teacher data."""
    A = stack.shape[1]
    idx = torch.arange(A)
    hold = idx % holdout_every == 0
    res = {}
    for order in range(1, max_order + 1):
        tr_angles = [a for a, h in zip(angles_deg, hold.tolist()) if not h]
        te_angles = [a for a, h in zip(angles_deg, hold.tolist()) if h]
        try:
            m = PSRF(tr_angles, order)
        except ValueError:
            continue
        coef = m.fit(stack[:, ~hold])
        pred = m.reconstruct(coef, te_angles)
        rmse = torch.sqrt(((pred - stack[:, hold]) ** 2).mean()).item()
        fit_tr = m.reconstruct(coef, tr_angles)
        rss = ((fit_tr - stack[:, ~hold]) ** 2).mean().item() + 1e-12
        n, k = len(tr_angles), 2 * order + 1
        res[order] = {"holdout_rmse": rmse, "bic": n * math.log(rss) + k * math.log(n)}
    res["best"] = min((o for o in res if o != "best"), key=lambda o: res[o]["holdout_rmse"]) if res else None
    return res
