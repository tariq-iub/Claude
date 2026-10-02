"""Latent optical field Z(x,y) and the differentiable physics layer that maps Z -> estimated Stokes (S_hat).

Z = [D(3), S(3), n(3), r, eta, k, rho, phi(2: cos2phi, sin2phi), g, u]  (17 channels, documented below)
    D   diffuse radiance (linear RGB)            S   specular radiance (linear RGB)
    n   unit normal (camera frame)               r   perceptual roughness in [0.05, 1]
    eta, k  *scale proxies* multiplying the base conductor constants (n0_c, k0_c) -> NOT physical constants
    rho, phi  optional learned residual polarization (magnitude, orientation) (gain 0 => pure physics)
    g   glare likelihood                         u   log-variance (aleatoric proxy)
Most of these are not identifiable from one RGB image; they are bounded proxies / distributions.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.normals import normalize
from ..optics.fresnel import fresnel_reflectance, preset
from ..optics.mueller import fresnel_mueller, rotate_mueller
from ..optics.reflection import diffuse_dolp, roughness_depolarization

Tensor = torch.Tensor

RAW_CH = 17          # D3 S3 n3 r eta k rho phi2 g
LOGVAR_CH = 6        # n_x, n_y, r, eta, k, split
BOUNDS = {"r": (0.05, 1.0), "eta": (0.5, 2.0), "k": (0.5, 1.5)}


def _bounded(x: Tensor, lo: float, hi: float) -> Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def decode_latent(raw: Tensor, normal_prior: Optional[Tensor] = None, prior_weight=None) -> Dict[str, Tensor]:
    """raw (B,17,H,W) unconstrained -> dict of physically bounded latent maps."""
    D = F.softplus(raw[:, 0:3])
    S = F.softplus(raw[:, 3:6])
    nxy, nz = raw[:, 6:8], F.softplus(raw[:, 8:9]) + 1e-2
    n = torch.cat([nxy, nz], 1)
    if normal_prior is not None and prior_weight is not None:
        n = n + prior_weight * normal_prior
    n = normalize(n)
    phi = raw[:, 13:15]
    phi = phi / phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return {
        "D": D, "S": S, "n": n,
        "r": _bounded(raw[:, 9:10], *BOUNDS["r"]),
        "eta": _bounded(raw[:, 10:11], *BOUNDS["eta"]),
        "k": _bounded(raw[:, 11:12], *BOUNDS["k"]),
        "rho": torch.sigmoid(raw[:, 12:13]),
        "phi2": phi,                       # (cos2phi, sin2phi)
        "g": torch.sigmoid(raw[:, 15:16]),
    }


class LatentOpticsHead(nn.Module):
    """1x1-conv head: features -> raw latent (17 ch) + log-variances (6 ch) over the identifiable-ish subset."""

    def __init__(self, in_ch: int, normal_prior_weight: float = 0.0):
        super().__init__()
        self.raw = nn.Conv2d(in_ch, RAW_CH, 1)
        self.logvar = nn.Conv2d(in_ch, LOGVAR_CH, 1)
        self.prior_weight = nn.Parameter(torch.tensor(float(normal_prior_weight)), requires_grad=normal_prior_weight > 0)
        nn.init.constant_(self.logvar.bias, -2.0)

    def forward(self, feat: Tensor, normal_prior: Optional[Tensor] = None) -> Dict[str, Tensor]:
        raw = self.raw(feat)
        lv = self.logvar(feat).clamp(-6, 3)
        out = decode_latent(raw, normal_prior, self.prior_weight if normal_prior is not None else None)
        out.update({"raw": raw, "logvar": lv, "u": lv.mean(1, keepdim=True), "normal_prior": normal_prior})
        return out

    def sample(self, out: Dict[str, Tensor], generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
        """Draw one physically plausible latent state Z^(k) by perturbing the uncertain variables with the predicted
        variances in *raw* space (so bounds are respected after decoding). D+S (total radiance) is preserved."""
        raw = out["raw"].clone()
        lv = out["logvar"]
        eps = lambda t: torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator)
        std = torch.exp(0.5 * lv)
        raw[:, 6:8] = raw[:, 6:8] + std[:, 0:2] * eps(std[:, 0:2])     # normal xy
        raw[:, 9:10] = raw[:, 9:10] + std[:, 2:3] * eps(std[:, 2:3])    # roughness
        raw[:, 10:11] = raw[:, 10:11] + std[:, 3:4] * eps(std[:, 3:4])  # eta proxy
        raw[:, 11:12] = raw[:, 11:12] + std[:, 4:5] * eps(std[:, 4:5])  # k proxy
        z = decode_latent(raw, out.get("normal_prior"), self.prior_weight if out.get("normal_prior") is not None else None)
        # diffuse/specular split: keep total, perturb logit of specular fraction
        T = (out["D"] + out["S"]).clamp_min(1e-6)
        f = (out["S"] / T).clamp(1e-4, 1 - 1e-4)
        f2 = torch.sigmoid(torch.logit(f) + std[:, 5:6] * eps(std[:, 5:6]))
        z["S"], z["D"] = T * f2, T * (1 - f2)
        z["rho"], z["phi2"], z["g"] = out["rho"], out["phi2"], out["g"]
        return z


class PhysicsStokesLayer(nn.Module):
    """Differentiable map  Z -> S_hat (B, 3 colour ch, 3 Stokes, H, W).

    Components (incoherent sum): specular  S_c [1, rho_s cos2psi, rho_s sin2psi]   (rho_s: conductor Fresnel DoLP x roughness
    depolarization), diffuse  D_c [1, rho_d cos2az, rho_d sin2az]   (Atkinson-Hancock, parallel to plane of emittance).
    illumination 'environment': theta_i = angle(n,v), psi = az(n) + pi/2.  'directional': Fresnel angle fixed by the
    half vector of (l, v), psi = azimuth(v x l)  (constant over the image for a distant light).
    """

    def __init__(self, base_material: str = "copper", illumination: str = "environment", residual_gain: float = 0.0,
                 r0: float = 0.5, learn_r0: bool = False, light_dir=(0.35, 0.25, 0.9)):
        super().__init__()
        n0, k0 = preset(base_material)
        self.register_buffer("n0", n0.view(1, 3, 1, 1))
        self.register_buffer("k0", k0.view(1, 3, 1, 1))
        self.illumination, self.residual_gain = illumination, residual_gain
        self.log_r0 = nn.Parameter(torch.tensor(math.log(r0)), requires_grad=learn_r0)
        self.light_dir = nn.Parameter(torch.tensor(light_dir, dtype=torch.float32), requires_grad=False)

    def forward(self, z: Dict[str, Tensor]) -> Tensor:
        n, D, S = z["n"], z["D"], z["S"]
        zen = torch.acos(n[:, 2:3].clamp(-1, 1))      # (B,1,H,W)
        az = torch.atan2(n[:, 1:2], n[:, 0:1])
        eta_c, k_c = self.n0 * z["eta"], self.k0 * z["k"]
        if self.illumination == "environment":
            theta, psi = zen, az + math.pi / 2
        elif self.illumination == "directional":
            l = F.normalize(self.light_dir, dim=0)
            h = F.normalize(l + torch.tensor([0.0, 0.0, 1.0], device=l.device), dim=0)
            theta = torch.acos(h[2].clamp(-1, 1)).expand_as(zen)
            sdir = torch.cross(torch.tensor([0.0, 0.0, 1.0], device=l.device), l, dim=0)
            psi = torch.atan2(sdir[1], sdir[0]).expand_as(zen)
        else:
            raise ValueError(self.illumination)
        Rs, Rp, _ = fresnel_reflectance(theta, eta_c, k_c)                 # (B,3,H,W)
        rho_s = (Rs - Rp) / (Rs + Rp).clamp_min(1e-9) * roughness_depolarization(z["r"], float(self.log_r0.exp()))
        n_d = (1.5 * z["eta"]).clamp_min(1.1)
        rho_d = diffuse_dolp(zen, n_d)                                      # (B,1,H,W)
        s1 = S * rho_s * torch.cos(2 * psi) + D * rho_d * torch.cos(2 * az)
        s2 = S * rho_s * torch.sin(2 * psi) + D * rho_d * torch.sin(2 * az)
        s0 = D + S
        if self.residual_gain > 0:
            mag = self.residual_gain * z["rho"] * s0
            s1 = s1 + mag * z["phi2"][:, 0:1]
            s2 = s2 + mag * z["phi2"][:, 1:2]
        # enforce realizability |(S1,S2)| <= S0
        m = torch.sqrt(s1 ** 2 + s2 ** 2 + 1e-12)
        scale = torch.minimum(torch.ones_like(m), s0 / m)
        return torch.stack([s0, s1 * scale, s2 * scale], dim=2)  # (B,C,3,H,W)

    def chain_intensity(self, z: Dict[str, Tensor], source_deg: float, analyzer_deg: float, eta_c: Optional[Tensor] = None,
                        k_c: Optional[Tensor] = None, n_diff: Optional[Tensor] = None) -> Tensor:
        """Virtual source-polarizer -> surface -> analyzer chain (cross/parallel-polarized imaging), (B,3,H,W).

        Specular term: full Mueller product M_A(theta_a) R(-psi) M_fresnel R(psi) S_src with S_src fully polarized at source_deg;
        the latent specular radiance S is rescaled by its mean Fresnel reflectance so that the *unpolarized-illumination*
        image is reproduced when the source is unpolarized. Diffuse term: depolarized by multiple scattering, so it keeps
        its (weak) Atkinson-Hancock polarization and ignores the source state (PROPOSED approximation: the diffuse light of a
        corrosion layer forgets the incident polarization). Cross-polarization therefore removes specular glare but keeps
        ~half of the diffuse signal.

        FRAME CONVENTION (open issue): source and analyzer angles are both expressed in the camera image-plane frame with the
        reflection Jones matrix diag(r_s, -r_p) (zero retardance at normal incidence). Unambiguous cases are tested (source aligned
        with s or p; normal incidence). For a source at 45 deg to the plane of incidence, the grazing-incidence phase flip
        (delta -> pi) maps +45 to -45 deg, so whether a physical "crossed" pair blocks the glare depends on how the two polarizer
        zeros are referenced across the mirror reflection. To be fixed by the hardware experiment before any quantitative use.
        """
        n, D, S = z["n"], z["D"], z["S"]
        zen = torch.acos(n[:, 2:3].clamp(-1, 1))
        az = torch.atan2(n[:, 1:2], n[:, 0:1])
        eta_c = self.n0 * z["eta"] if eta_c is None else eta_c        # absolute constants may be supplied (validation with true optics)
        k_c = self.k0 * z["k"] if k_c is None else k_c
        if self.illumination == "environment":
            theta, psi = zen, az + math.pi / 2
        else:
            l = F.normalize(self.light_dir, dim=0)
            h = F.normalize(l + torch.tensor([0.0, 0.0, 1.0], device=l.device), dim=0)
            theta = torch.acos(h[2].clamp(-1, 1)).expand_as(zen)
            sdir = torch.cross(torch.tensor([0.0, 0.0, 1.0], device=l.device), l, dim=0)
            psi = torch.atan2(sdir[1], sdir[0]).expand_as(zen)
        Rs, Rp, delta = fresnel_reflectance(theta, eta_c, k_c)
        M = rotate_mueller(fresnel_mueller(Rs, Rp, delta), psi.expand_as(Rs))          # (B,3,H,W,4,4) image frame
        a = math.radians(source_deg)
        S_src = torch.tensor([1.0, math.cos(2 * a), math.sin(2 * a), 0.0], device=M.device, dtype=M.dtype)
        out = torch.einsum("...ij,j->...i", M, S_src)                                  # (B,3,H,W,4)
        b = math.radians(analyzer_deg)
        row = 0.5 * torch.tensor([1.0, math.cos(2 * b), math.sin(2 * b), 0.0], device=M.device, dtype=M.dtype)
        I_spec_unit = (out * row).sum(-1)                                              # per unit incident irradiance
        F0 = (0.5 * (Rs + Rp)).clamp_min(1e-6)
        I_spec = S / F0 * I_spec_unit
        rho_d = diffuse_dolp(zen, (1.5 * z["eta"]).clamp_min(1.1) if n_diff is None else n_diff)
        I_diff = 0.5 * D * (1 + rho_d * torch.cos(2 * (az - b)))
        return I_spec + I_diff
