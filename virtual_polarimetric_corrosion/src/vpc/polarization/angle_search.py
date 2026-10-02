"""Automatic virtual-analyzer optimisation: theta* = argmax_theta J(theta).

J(theta) = alpha T + beta C + gamma E - delta G - lambda U with each term min-max normalised over theta so the
weights are comparable. The curve is interpolated with a *periodic* PCHIP (period 180 deg; monotone, no overshoot; a periodic
cubic spline is available via interp='periodic_spline'); the measured (sampled) points are always returned alongside the fitted curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import laplace, sobel


@dataclass
class AngleSearchResult:
    angles_deg: np.ndarray
    terms: dict            # raw (un-normalised) per-angle terms: T, C, E, G, U
    terms_norm: dict       # min-max normalised terms
    J: np.ndarray          # objective at sampled angles
    fine_angles_deg: np.ndarray
    J_fit: np.ndarray      # periodic cubic spline
    theta_star_deg: float
    theta_star_sampled_deg: float


def _minmax(x: np.ndarray) -> np.ndarray:
    r = x.max() - x.min()
    return np.zeros_like(x) if r < 1e-12 else (x - x.min()) / r


def angle_terms(stack: np.ndarray, mask: Optional[np.ndarray] = None, uncertainty: Optional[np.ndarray] = None,
                glare_thresh: float = 0.95) -> dict:
    """stack (A,H,W) luminance analyzer images in [0,1]; mask (H,W) bool corrosion region (predicted or annotated);
    uncertainty (A,H,W) per-angle std of the CPE ensemble (optional).
    T: texture preservation = laplacian-energy(I_theta) / mean-over-angles laplacian-energy (high-frequency kept)
    C: Fisher-style contrast |mu_c - mu_b| / (sigma_c + sigma_b) between corrosion and background (needs mask)
    E: mean gradient magnitude (edge visibility)
    G: glare energy = mean(max(I - glare_thresh, 0)) + saturated-area fraction
    U: mean uncertainty."""
    A = stack.shape[0]
    T, C, E, G, U = (np.zeros(A) for _ in range(5))
    for i in range(A):
        im = stack[i]
        T[i] = np.mean(laplace(im) ** 2)
        gx, gy = sobel(im, axis=1) / 8.0, sobel(im, axis=0) / 8.0
        E[i] = np.mean(np.hypot(gx, gy))
        G[i] = np.mean(np.maximum(im - glare_thresh, 0)) + np.mean(im >= glare_thresh)
        if mask is not None and mask.any() and (~mask).any():
            a, b = im[mask], im[~mask]
            C[i] = abs(a.mean() - b.mean()) / (a.std() + b.std() + 1e-6)
        if uncertainty is not None:
            U[i] = uncertainty[i].mean()
    T = T / (T.mean() + 1e-12)
    return {"T": T, "C": C, "E": E, "G": G, "U": U}


def optimise_analyzer(stack: np.ndarray, angles_deg, mask=None, uncertainty=None,
                      weights=(1.0, 1.0, 1.0, 1.0, 1.0), fine_step: float = 0.5, glare_thresh: float = 0.95,
                      interp: str = "pchip") -> AngleSearchResult:
    angles = np.asarray(angles_deg, dtype=float)
    terms = angle_terms(stack, mask, uncertainty, glare_thresh)
    norm = {k: _minmax(v) for k, v in terms.items()}
    a, b, g, d, l = weights
    J = a * norm["T"] + b * norm["C"] + g * norm["E"] - d * norm["G"] - l * norm["U"]
    order = np.argsort(angles % 180.0)
    xa, ya = angles[order] % 180.0, J[order]
    fine = np.arange(0.0, 180.0, fine_step)
    if interp == "pchip":
        # periodic PCHIP: wrap the samples one period each side. Monotone between nodes => the fitted curve can never invent an
        # extremum between measured points (a cubic spline can overshoot); theta* is then resolved only up to the sampling grid.
        xx = np.concatenate([xa - 180.0, xa, xa + 180.0]); yy = np.tile(ya, 3)
        Jf = PchipInterpolator(xx, yy)(fine)
    elif interp == "periodic_spline":
        x = np.append(xa, 180.0 + xa[0]); y = np.append(ya, ya[0])
        Jf = CubicSpline(x, y, bc_type="periodic")(fine)
    else:
        raise ValueError(interp)
    return AngleSearchResult(angles, terms, norm, J, fine, Jf, float(fine[np.argmax(Jf)]), float(angles[np.argmax(J)] % 180.0))
