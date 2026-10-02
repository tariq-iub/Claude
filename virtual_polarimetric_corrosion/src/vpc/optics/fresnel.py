"""Fresnel reflection for dielectrics and conductors (complex refractive index).

Conventions (documented because sign conventions differ between textbooks):
  * time dependence exp(-i w t); complex index N = n + i k with k >= 0 (absorbing medium);
  * s-polarization = E perpendicular to the plane of incidence, p = parallel;
  * amplitude coefficients
        r_s = (cos_i - N cos_t) / (cos_i + N cos_t)
        r_p = (N cos_i - cos_t) / (N cos_i + cos_t)
    with Snell cos_t = sqrt(1 - sin_i^2 / N^2) (principal branch, complex for metals and TIR);
  * With these, r_p = -r_s at normal incidence (a pure sign convention). The *Jones reflection matrix*
    used in mueller.py is diag(r_s, -r_p) so that it equals r_s * Identity at normal incidence.
  * relative retardance delta = arg(-r_p) - arg(r_s), which is 0 at normal incidence.

Established physics: Born & Wolf, Principles of Optics; Hecht, Optics; Chipman, Lam & Young,
Polarized Light and Optical Systems. Approximations (flagged): the "RGB-effective" optical constants
below replace wavelength-integrated, sensor-weighted values; corrosion layers are treated as semi-infinite
homogeneous media (no thin-film interference).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

Tensor = torch.Tensor


def _as_tensor(x, like: Tensor | None = None) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=(like.dtype if like is not None and like.is_floating_point() else torch.float32))


def fresnel_amplitudes(theta_i, n, k=0.0) -> Tuple[Tensor, Tensor]:
    """Complex amplitude reflection coefficients (r_s, r_p) for incidence from vacuum/air onto N = n + ik.

    ``theta_i`` in radians (incidence angle from the surface normal). ``n``, ``k`` broadcast with theta_i.
    For a dielectric (k=0) with relative index n = n2/n1 this is the classical Fresnel result and handles
    total internal reflection (n<1) through the complex square root.
    """
    theta_i = _as_tensor(theta_i)
    dt = theta_i.dtype if theta_i.is_floating_point() else torch.float32
    theta_i = theta_i.to(dt)
    n = _as_tensor(n, theta_i).to(dt)
    k = _as_tensor(k, theta_i).to(dt)
    n, k = torch.broadcast_tensors(n, k)
    N = torch.complex(n, k)
    ci = torch.cos(theta_i).to(N.dtype)
    si = torch.sin(theta_i).to(N.dtype)
    ct = torch.sqrt(1.0 - (si / N) ** 2)
    rs = (ci - N * ct) / (ci + N * ct)
    rp = (N * ci - ct) / (N * ci + ct)
    return rs, rp


def fresnel_reflectance(theta_i, n, k=0.0) -> Tuple[Tensor, Tensor, Tensor]:
    """Return (R_s, R_p, delta): intensity reflectances |r|^2 and relative retardance (radians)."""
    rs, rp = fresnel_amplitudes(theta_i, n, k)
    Rs, Rp = rs.abs() ** 2, rp.abs() ** 2
    delta = torch.angle(-rp) - torch.angle(rs)
    delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    return Rs, Rp, delta


def dielectric_fresnel(theta_i, n1=1.0, n2=1.5) -> Tuple[Tensor, Tensor]:
    """R_s, R_p for a lossless dielectric interface n1 -> n2 (real indices)."""
    Rs, Rp, _ = fresnel_reflectance(theta_i, _as_tensor(n2) / _as_tensor(n1), 0.0)
    return Rs, Rp


def conductor_fresnel(theta_i, n, k) -> Tuple[Tensor, Tensor, Tensor]:
    """R_s, R_p, delta for a conductor with complex index n + ik (e.g. copper)."""
    return fresnel_reflectance(theta_i, n, k)


def fresnel_dolp(theta_i, n, k=0.0) -> Tensor:
    """Degree of linear polarization of a specular reflection of UNPOLARIZED light:
    (R_s - R_p) / (R_s + R_p). Positive: s-dominant (perpendicular to plane of incidence)."""
    Rs, Rp, _ = fresnel_reflectance(theta_i, n, k)
    return (Rs - Rp) / (Rs + Rp).clamp_min(1e-12)


def brewster_angle(n2: float, n1: float = 1.0) -> float:
    """Brewster angle for a lossless dielectric (radians)."""
    return math.atan2(n2, n1)


def principal_angle_of_incidence(n: float, k: float, num: int = 4001) -> float:
    """Angle (rad) of minimum R_p for a conductor (pseudo-Brewster). Numerical search."""
    th = torch.linspace(0, math.pi / 2 - 1e-4, num, dtype=torch.float64)
    _, Rp, _ = fresnel_reflectance(th, n, k)
    return float(th[Rp.argmin()])


# ---------------------------------------------------------------------------------------------
# Optical-constant presets.  ALL VALUES BELOW ARE ILLUSTRATIVE PROXIES, NOT MEASUREMENTS OF THE
# INSPECTED CARTRIDGES.  Replace with sensor-weighted integrals of tabulated spectra
# (e.g. Johnson & Christy, Phys. Rev. B 6, 4370 (1972)) once the camera spectral response is known.
# ---------------------------------------------------------------------------------------------
# (n_R, n_G, n_B), (k_R, k_G, k_B): widely used RGB-effective copper constants from rendering
# literature (PBRT/Mitsuba-style tables). Treat as order-of-magnitude, wavelength-trend-correct.
COPPER_RGB = {"n": (0.200, 0.924, 1.102), "k": (3.912, 2.452, 2.142)}
GOLD_RGB = {"n": (0.143, 0.375, 1.442), "k": (3.983, 2.386, 1.603)}
# Bronze/brass proxy: midpoint of Cu and Au tables (alloys are redder-yellow than Cu). Pure assumption.
BRONZE_PROXY_RGB = {
    "n": tuple(0.5 * (a + b) for a, b in zip(COPPER_RGB["n"], GOLD_RGB["n"])),
    "k": tuple(0.5 * (a + b) for a, b in zip(COPPER_RGB["k"], GOLD_RGB["k"])),
}
# Corrosion-product proxies: weakly absorbing dielectric-like layers. Ranges, not measurements.
CU2O_PROXY_RGB = {"n": (2.9, 3.0, 3.2), "k": (0.15, 0.30, 0.60)}   # cuprite-like (dark red/brown)
CUO_PROXY_RGB = {"n": (2.6, 2.6, 2.6), "k": (0.50, 0.50, 0.50)}    # tenorite-like (black)
PATINA_PROXY_RGB = {"n": (1.75, 1.75, 1.75), "k": (0.02, 0.01, 0.01)}  # malachite/brochantite-like (green)
CHLORIDE_PROXY_RGB = {"n": (1.70, 1.70, 1.70), "k": (0.01, 0.01, 0.01)}  # atacamite/paratacamite-like

OPTICAL_PRESETS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "copper": COPPER_RGB,
    "bronze_proxy": BRONZE_PROXY_RGB,
    "cu2o_like": CU2O_PROXY_RGB,
    "cuo_like": CUO_PROXY_RGB,
    "patina": PATINA_PROXY_RGB,
    "chloride": CHLORIDE_PROXY_RGB,
}


def preset(name: str, dtype=torch.float32) -> Tuple[Tensor, Tensor]:
    """Return (n, k) tensors of shape (3,) for an optical preset (R, G, B order)."""
    p = OPTICAL_PRESETS[name]
    return torch.tensor(p["n"], dtype=dtype), torch.tensor(p["k"], dtype=dtype)
