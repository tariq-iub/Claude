"""Polarimetric reflection model: I = I_diffuse + I_specular + I_scatter, with Stokes bookkeeping.

Established physics: Fresnel (conductor/dielectric), GGX microfacet lobe, Lambertian-like diffuse base,
Atkinson & Hancock diffuse polarization (light that is refracted out of a dielectric layer is partially
polarized *parallel* to the plane of emittance).
Proposed approximations (flagged):
  * environment-illumination mode: mirror-like specular reflection from an extended source samples the
    environment around the mirror direction, so theta_i = angle(n, v) and AoLP_spec = azimuth(n) + pi/2;
  * roughness depolarization rho_eff = rho_fresnel * exp(-(r/r0)^2) (smooth interpolation, r0 empirical);
  * scatter term is fully depolarized;
  * components add incoherently (valid: Stokes vectors of mutually incoherent beams add).
Nothing here assumes diffuse = unpolarized or specular = fully polarized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch

from .fresnel import fresnel_reflectance
from .microfacet import specular_stokes_cook_torrance, _normalize

Tensor = torch.Tensor


def diffuse_dolp(theta: Tensor, n: Tensor | float = 1.5) -> Tensor:
    """Atkinson-Hancock (2006) diffuse degree of polarization vs emittance zenith angle theta (radians),
    for dielectric refractive index n (>1). Zero at theta=0, increases towards grazing view."""
    n = torch.as_tensor(n, dtype=theta.dtype, device=theta.device)
    s2 = torch.sin(theta) ** 2
    num = (n - 1.0 / n) ** 2 * s2
    den = 2 + 2 * n ** 2 - (n + 1.0 / n) ** 2 * s2 + 4 * torch.cos(theta) * torch.sqrt((n ** 2 - s2).clamp_min(1e-9))
    return (num / den).clamp(0, 1)


def roughness_depolarization(r: Tensor, r0: float = 0.5) -> Tensor:
    """PROPOSED empirical attenuation of specular DoLP with roughness (r in [0,1])."""
    return torch.exp(-(r / r0) ** 2)


def zenith_azimuth(n: Tensor, dim: int = -1) -> tuple[Tensor, Tensor]:
    """Zenith (angle to view axis z) and azimuth of normals; camera frame x right, y up, z toward camera."""
    nz = n.select(dim, 2).clamp(-1, 1)
    return torch.acos(nz), torch.atan2(n.select(dim, 1), n.select(dim, 0))


def specular_stokes_environment(n: Tensor, eta: Tensor, kappa: Tensor, rough: Tensor, r0: float = 0.5) -> Tensor:
    """Per-channel specular Stokes *fractions* (S0=Fresnel mean reflectance, S1, S2) for mirror-like reflection
    under extended illumination. n (...,3); eta, kappa (...,C); returns (...,C,3)."""
    zen, az = zenith_azimuth(n)
    Rs, Rp, _ = fresnel_reflectance(zen.unsqueeze(-1), eta, kappa)
    F0, dF = 0.5 * (Rs + Rp), 0.5 * (Rs - Rp) * roughness_depolarization(rough, r0).unsqueeze(-1)
    psi = (az + math.pi / 2).unsqueeze(-1)  # s-axis is perpendicular to the plane containing n and v
    return torch.stack([F0, dF * torch.cos(2 * psi), dF * torch.sin(2 * psi)], dim=-1)


def diffuse_stokes(D: Tensor, n: Tensor, n_diff: Tensor | float = 1.5) -> Tensor:
    """Per-channel diffuse Stokes from diffuse radiance D (...,C): S0=D, polarized part parallel to plane of
    emittance, i.e. AoLP = azimuth(n). Returns (...,C,3)."""
    zen, az = zenith_azimuth(n)
    rho = diffuse_dolp(zen, n_diff).unsqueeze(-1)
    c2, s2 = torch.cos(2 * az).unsqueeze(-1), torch.sin(2 * az).unsqueeze(-1)
    return torch.stack([D, D * rho * c2, D * rho * s2], dim=-1)


@dataclass
class RenderInputs:
    albedo: Tensor      # (H,W,C) diffuse albedo in linear RGB
    normals: Tensor     # (H,W,3)
    rough: Tensor       # (H,W) perceptual roughness
    eta: Tensor         # (H,W,C)
    kappa: Tensor       # (H,W,C)
    spec_weight: Tensor  # (H,W) multiplies the specular term (porosity/coverage)
    diff_weight: Tensor  # (H,W) multiplies diffuse term
    scatter: Tensor      # (H,W,C) unpolarized additive term (volume scatter / ambient)
    n_diff: Tensor       # (H,W) refractive index controlling diffuse polarization


def render_stokes(
    inp: RenderInputs, light_dir: Tensor, view_dir: Tensor | None = None, irradiance: float = 1.0,
    mode: str = "environment", env_strength: float = 1.0, r0: float = 0.5,
) -> Dict[str, Tensor]:
    """Render per-channel linear Stokes S (H,W,C,3) and the three components.

    mode: 'directional' (point/distant light, GGX lobe, Fresnel at the half-angle) or
          'environment' (extended/ambient source, mirror-like sampling; Fresnel at angle(n,v)).
    The directional case also adds a weak mirror 'environment' term only if env_strength>0 and mode='directional'
    is not intended; keep env_strength=0 there.
    """
    n = _normalize(inp.normals)
    v = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype, device=n.device) if view_dir is None else view_dir
    l = _normalize(light_dir.to(n))
    cos_nl = (n * l).sum(-1).clamp_min(0)
    D = inp.diff_weight.unsqueeze(-1) * inp.albedo * cos_nl.unsqueeze(-1) * irradiance / math.pi
    S_diff = diffuse_stokes(D, n, inp.n_diff)
    if mode == "directional":
        S_spec, _ = specular_stokes_cook_torrance(n, l.expand_as(n), v.expand_as(n), inp.rough, inp.eta, inp.kappa)
        S_spec = S_spec * irradiance
    elif mode == "environment":
        S_spec = specular_stokes_environment(n, inp.eta, inp.kappa, inp.rough, r0) * env_strength
        # roughness spreads the mirror lobe: effective environment radiance is smoothed; amplitude kept (documented)
    else:
        raise ValueError(mode)
    S_spec = S_spec * inp.spec_weight.unsqueeze(-1).unsqueeze(-1)
    S_sc = torch.stack([inp.scatter, torch.zeros_like(inp.scatter), torch.zeros_like(inp.scatter)], dim=-1)
    return {"S": S_diff + S_spec + S_sc, "S_diffuse": S_diff, "S_specular": S_spec, "S_scatter": S_sc}
