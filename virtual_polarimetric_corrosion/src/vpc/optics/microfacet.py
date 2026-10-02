"""GGX / Trowbridge-Reitz microfacet specular lobe with polarized (Mueller) Fresnel term.

Validity: microfacet theory assumes geometric optics (roughness correlation length >> wavelength),
single scattering, and an isotropic height distribution. It is a reasonable model for polished and
moderately rough metal, and a weak one for (i) powdery/porous corrosion deposits (volume scattering),
(ii) pits whose size approaches the shadowing/masking length scale, (iii) thin oxide films (interference).
"""
from __future__ import annotations

import math

import torch

from .fresnel import fresnel_reflectance
from .mueller import fresnel_mueller, rotate_mueller, apply_mueller

Tensor = torch.Tensor
_EPS = 1e-6


def _normalize(v: Tensor, dim: int = -1) -> Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp_min(_EPS)


def ggx_D(cos_h: Tensor, alpha: Tensor) -> Tensor:
    """Trowbridge-Reitz normal distribution; alpha = roughness^2 (perceptual roughness r -> alpha=r^2)."""
    a2 = alpha ** 2
    d = cos_h ** 2 * (a2 - 1.0) + 1.0
    return a2 / (math.pi * d ** 2 + _EPS)


def smith_G1(cos_x: Tensor, alpha: Tensor) -> Tensor:
    a2 = alpha ** 2
    c2 = cos_x.clamp_min(_EPS) ** 2
    return 2.0 * cos_x.clamp_min(_EPS) / (cos_x.clamp_min(_EPS) + torch.sqrt(a2 + (1 - a2) * c2))


def smith_G(cos_l: Tensor, cos_v: Tensor, alpha: Tensor) -> Tensor:
    return smith_G1(cos_l, alpha) * smith_G1(cos_v, alpha)


def specular_stokes_cook_torrance(
    n: Tensor, l: Tensor, v: Tensor, roughness: Tensor, eta: Tensor, kappa: Tensor,
    S_in: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Polarized Cook-Torrance specular radiance factor (per unit irradiance on the horizontal facet).

    Shapes: n (...,3) unit normals, l, v (...,3) unit light/view directions (broadcast), roughness (...),
    eta, kappa (...,C) per-channel complex index parts. Returns (S, theta_d) where S is
    (...,C,3) = [S0, S1, S2] in the image frame (x,y perpendicular to v) and theta_d the Fresnel angle.

    Physics: microfacet normal = half vector h; Fresnel angle theta_d = angle(h, v) (independent of n for
    a distant light); plane of incidence contains l, v, h; the s axis is v x l, so for unpolarized light the
    specular AoLP is the azimuth of (v x l). Unpolarized input unless ``S_in`` (...,4) is given.
    """
    h = _normalize(l + v)
    cos_nl = (n * l).sum(-1).clamp_min(0)
    cos_nv = (n * v).sum(-1).clamp_min(_EPS)
    cos_nh = (n * h).sum(-1).clamp(0, 1)
    cos_d = (h * v).sum(-1).clamp(0, 1)
    theta_d = torch.acos(cos_d)
    alpha = (roughness.clamp(0.02, 1.0)) ** 2
    Dg = ggx_D(cos_nh, alpha) * smith_G(cos_nl, cos_nv, alpha) / (4.0 * cos_nv)  # radiance factor (cos_l folded)
    Rs, Rp, delta = fresnel_reflectance(theta_d.unsqueeze(-1), eta, kappa)  # (...,C)
    # s-axis direction in image frame
    s_dir = _normalize(torch.cross(v.expand_as(l), l.expand_as(v), dim=-1))
    psi = torch.atan2(s_dir[..., 1], s_dir[..., 0])  # (...)
    if S_in is None:
        F0, dF = 0.5 * (Rs + Rp), 0.5 * (Rs - Rp)
        c2, s2 = torch.cos(2 * psi).unsqueeze(-1), torch.sin(2 * psi).unsqueeze(-1)
        S = torch.stack([F0, dF * c2, dF * s2], dim=-1)
    else:  # full Mueller path (polarized illumination)
        M_pl = fresnel_mueller(Rs, Rp, delta)  # (...,C,4,4)
        M_img = rotate_mueller(M_pl, psi.unsqueeze(-1))
        Sout = apply_mueller(M_img, S_in.unsqueeze(-2).expand(*M_img.shape[:-2], 4))
        S = Sout[..., :3]
    return S * Dg.unsqueeze(-1).unsqueeze(-1), theta_d
