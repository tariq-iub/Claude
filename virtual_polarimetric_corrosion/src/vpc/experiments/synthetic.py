"""Synthetic polarimetric renderer for cylindrical copper/bronze cartridges (SYNTHETIC VALIDATION ONLY).

Purpose: generate controlled ground truth (normals, roughness, optical constants, S0..S2, DoLP, AoLP, analyzer images
I_theta) to *unit-validate the virtual-physics modules* and to pre-train under domain randomization. All material
parameters are illustrative proxies, never measurements; results on this data MUST NOT be reported as corrosion
performance on real cartridges (docs/THREATS_TO_VALIDITY.md).

A cartridge identity ("group") fixes the corrosion texture in cylinder coordinates (axial u, circumferential theta);
each *view* varies axis orientation, roll, lighting, exposure, white balance and noise. Group-level splitting is therefore
testable (tests/test_splits.py, scripts/check_leakage.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..color import linear_to_srgb
from ..corrosion.segmentation import CORRODED, NUM_CLASSES
from ..corrosion.severity import DEFAULT_BINS
from ..geometry.cylinder import cylinder_normals, fit_cylinder_from_mask
from ..optics.fresnel import OPTICAL_PRESETS
from ..optics.reflection import RenderInputs, render_stokes
from ..polarization.virtual_analyzer import CANONICAL_ANGLES_DEG

Tensor = torch.Tensor

# per-class illustrative appearance (index = class id). Fields: albedo(3), diff_w, spec_w, rough, scatter, bump(normal noise), optical preset
_CLASS_TABLE = {
    0: dict(albedo=(0.10, 0.10, 0.11), diff=1.0, spec=0.02, rough=0.9, scatter=0.0, bump=0.0, optics="chloride"),      # background (replaced below)
    1: dict(albedo=None, diff=0.12, spec=1.0, rough=0.20, scatter=0.01, bump=0.01, optics=None),                          # healthy: substrate
    2: dict(albedo=(0.28, 0.14, 0.08), diff=0.45, spec=0.80, rough=0.30, scatter=0.01, bump=0.03, optics="substrate_tarnish"),
    3: dict(albedo=(0.14, 0.07, 0.05), diff=0.90, spec=0.35, rough=0.45, scatter=0.01, bump=0.08, optics="cu2o_like"),
    4: dict(albedo=(0.10, 0.30, 0.22), diff=1.00, spec=0.25, rough=0.55, scatter=0.01, bump=0.10, optics="patina"),
    5: dict(albedo=(0.45, 0.58, 0.50), diff=1.10, spec=0.08, rough=0.85, scatter=0.03, bump=0.20, optics="chloride"),
    6: dict(albedo=(0.03, 0.025, 0.02), diff=0.60, spec=0.10, rough=0.70, scatter=0.0, bump=0.10, optics="cuo_like"),
}
_SUBSTRATE_ALBEDO = {"copper": (0.60, 0.28, 0.18), "bronze_proxy": (0.55, 0.38, 0.15)}


@dataclass
class SynthConfig:
    size: int = 96
    angles_deg: Sequence[float] = CANONICAL_ANGLES_DEG
    p_environment: float = 0.6        # fraction of views under extended/environment illumination
    p_pitting: float = 0.5
    exposure_range: tuple = (0.6, 1.6)   # target p99.5 of S0 on the object (>1 => clipping/glare)
    read_noise: float = 0.004
    shot_noise: float = 0.01
    domain_randomization: bool = True
    sensor_clip: bool = True          # False: keep radiance unclipped (exact-identity validation only; real sensors clip)
    texture_axial: int = 160
    texture_circ: int = 192


def _spectral_noise(rng: np.random.Generator, h: int, w: int, beta: float = 2.2) -> np.ndarray:
    """Periodic 1/f^beta noise in [0,1] (tileable, so circumferential wrap is seamless)."""
    f = np.fft.fft2(rng.standard_normal((h, w)))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    x = np.real(np.fft.ifft2(f / r ** (beta / 2)))
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return x


def make_group(seed: int, cfg: SynthConfig, level: Optional[float] = None) -> Dict:
    """Cartridge identity: canonical corrosion label map, pit height map, substrate, albedo/roughness jitter."""
    rng = np.random.default_rng(seed)
    na, nt = cfg.texture_axial, cfg.texture_circ
    level = float(rng.uniform(0, 1) if level is None else level)
    fields = [_spectral_noise(rng, na, nt, b) for b in (2.6, 2.4, 2.2, 2.0)]
    label = np.ones((na, nt), np.int64)
    fr = [0.05 + 0.5 * level * rng.uniform(0.4, 1.0), 0.4 * level ** 1.3 * rng.uniform(0, 1),
          0.35 * level * rng.uniform(0, 1), 0.15 * level ** 2 * rng.uniform(0, 1)]
    for cls, f, frac in zip((2, 3, 4, 5), fields, fr):
        if frac > 0.005:
            label[f > np.quantile(f, 1 - frac)] = cls
    height = np.zeros((na, nt), np.float32)
    pitted = rng.uniform() < cfg.p_pitting and level > 0.15
    if pitted:
        n_pits = rng.poisson(10 + 60 * level ** 1.2)
        yy, xx = np.mgrid[:na, :nt]
        for _ in range(n_pits):
            cy, cx, rad = rng.uniform(0, na), rng.uniform(0, nt), rng.uniform(1.2, 3.2)
            d2 = (yy - cy) ** 2 + (((xx - cx + nt / 2) % nt) - nt / 2) ** 2
            height -= np.exp(-d2 / (2 * (rad * 0.7) ** 2)).astype(np.float32)
            label[d2 < rad ** 2] = 6
    return dict(seed=seed, level=level, label=label, height=height, substrate=str(rng.choice(["copper", "bronze_proxy"])),
                albedo_jit=np.exp(rng.normal(0, 0.12, 3)), rough_jit=float(np.exp(rng.normal(0, 0.25))),
                pitted=bool(pitted), finish_noise=_spectral_noise(rng, na, nt, 3.0))


def _class_params(substrate: str, albedo_jit: np.ndarray, rough_jit: float, dr: bool, rng) -> Dict[str, np.ndarray]:
    K = NUM_CLASSES
    albedo, diff, spec, rough, scatter, bump = (np.zeros((K, 3)), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K))
    eta, kap = np.zeros((K, 3)), np.zeros((K, 3))
    for c, p in _CLASS_TABLE.items():
        albedo[c] = _SUBSTRATE_ALBEDO[substrate] if p["albedo"] is None else p["albedo"]
        diff[c], spec[c], rough[c], scatter[c], bump[c] = p["diff"], p["spec"], p["rough"], p["scatter"], p["bump"]
        name = p["optics"]
        if name is None:
            name = substrate
        if name == "substrate_tarnish":
            n, k = np.array(OPTICAL_PRESETS[substrate]["n"]) * 1.2, np.array(OPTICAL_PRESETS[substrate]["k"]) * 0.85
        else:
            n, k = np.array(OPTICAL_PRESETS[name]["n"]), np.array(OPTICAL_PRESETS[name]["k"])
        eta[c], kap[c] = n, k
    if dr:
        albedo = albedo * albedo_jit * np.exp(rng.normal(0, 0.08, (K, 3)))
        rough = np.clip(rough * rough_jit * np.exp(rng.normal(0, 0.15, K)), 0.05, 1.0)
        eta = eta * np.exp(rng.normal(0, 0.05, (K, 3)))
        kap = kap * np.exp(rng.normal(0, 0.05, (K, 3)))
    return dict(albedo=albedo, diff=diff, spec=spec, rough=rough, scatter=scatter, bump=bump, eta=eta, kappa=kap)


def _env_radiance(n: Tensor, rng, rough: Tensor) -> Tensor:
    """Extended-source environment seen through mirror-like reflection: ambient + 1-3 soft boxes. n (H,W,3)."""
    v = torch.tensor([0, 0, 1.0])
    r = 2 * (n * v).sum(-1, keepdim=True) * n - v
    E = torch.full(r.shape[:2], float(rng.uniform(0.03, 0.12)))
    for _ in range(int(rng.integers(1, 4))):
        d = torch.tensor([rng.normal(), rng.normal(), abs(rng.normal()) + 0.5], dtype=torch.float32)
        d = d / d.norm()
        kappa = float(rng.uniform(6, 60))
        kappa_eff = 1.0 / (1.0 / kappa + 4 * rough ** 4)          # roughness blurs the reflected softbox
        E = E + float(rng.uniform(0.6, 2.5)) * torch.exp(kappa_eff * ((r * d).sum(-1) - 1.0))
    return E


def render_view(group: Dict, seed: int, cfg: SynthConfig, force_mode: Optional[str] = None) -> Dict:
    rng = np.random.default_rng(seed)
    H = W = cfg.size
    dr = cfg.domain_randomization
    alpha = float(rng.uniform(0, math.pi))
    R = H * float(rng.uniform(0.20, 0.30))
    L = H * float(rng.uniform(0.75, 0.95))
    cx, cy = W / 2 + rng.uniform(-0.05, 0.05) * W, H / 2 + rng.uniform(-0.05, 0.05) * H
    elev = float(rng.uniform(-0.15, 0.15)) if dr else 0.0
    roll = float(rng.uniform(0, 2 * math.pi))
    ys, xs = np.mgrid[:H, :W].astype(np.float64)
    x, yup = xs - cx, -ys + cy
    ca, sa = math.cos(alpha), math.sin(alpha)
    ax, lat = x * ca + yup * sa, -x * sa + yup * ca
    obj = (np.abs(lat) <= R) & (np.abs(ax) <= L / 2)
    s = np.clip(lat / R, -1, 1)
    theta = np.arcsin(s)
    na, nt = group["label"].shape
    iu = np.clip(((ax / L + 0.5) * (na - 1)).round().astype(int), 0, na - 1)
    it = (((theta + roll) / (2 * math.pi)) % 1.0 * nt).astype(int) % nt
    label = np.where(obj, group["label"][iu, it], 0)
    height = np.where(obj, group["height"][iu, it], 0.0)
    finish = np.where(obj, group["finish_noise"][iu, it], 0.5)

    n_true, _ = cylinder_normals(H, W, alpha, cx, -cy, R, elev)
    n = n_true.permute(1, 2, 0).numpy().astype(np.float64)
    prm = _class_params(group["substrate"], group["albedo_jit"], group["rough_jit"], dr, rng)
    # bump + pit-bowl normal perturbation
    bump_amp = prm["bump"][label]
    noise = _spectral_noise(rng, H, W, 1.6) - 0.5
    gy_, gx_ = np.gradient(height)
    n[..., 0] += bump_amp * 2 * noise - 0.9 * gx_ * obj
    n[..., 1] += bump_amp * 2 * (np.roll(noise, 7, 0)) - 0.9 * (-gy_) * obj
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(obj[..., None], n, np.array([0, 0, 1.0]))

    to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    rough = prm["rough"][label] * (1 + 0.25 * (finish - 0.5)) if dr else prm["rough"][label]
    rough = np.clip(rough, 0.05, 1.0)
    albedo = prm["albedo"][label]
    bg = rng.uniform(0.04, 0.35)
    grad = 0.6 + 0.4 * (xs / W) * rng.uniform(-1, 1) + 0.1 * (_spectral_noise(rng, H, W, 2.5) - 0.5)
    scatter = np.where(obj[..., None], prm["scatter"][label][..., None], (bg * grad)[..., None] * np.ones(3))
    spec_w = prm["spec"][label].copy()
    diff_w = prm["diff"][label].copy()
    pit_occl = np.where(label == 6, 0.5, 1.0)
    diff_w *= pit_occl
    n_diff = np.where(label == 1, 1.8, 1.7)
    bgm = ~obj
    spec_w[bgm], diff_w[bgm], albedo[bgm] = 0.0, 0.0, 0.0   # background is purely unpolarized scatter

    mode = force_mode or ("environment" if rng.uniform() < cfg.p_environment else "directional")
    nt_, rough_t = to_t(n), to_t(rough)
    if mode == "environment":
        E = _env_radiance(nt_, rng, rough_t)
        spec_w_t = to_t(spec_w) * E
        light = torch.tensor([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0])
        light = light / light.norm()
        irr = float(rng.uniform(0.8, 1.6))
    else:
        light = torch.tensor([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.4, 1.0)])
        light = light / light.norm()
        spec_w_t, irr = to_t(spec_w), float(rng.uniform(0.8, 2.5))
    inp = RenderInputs(albedo=to_t(albedo), normals=nt_, rough=rough_t, eta=to_t(prm["eta"][label]), kappa=to_t(prm["kappa"][label]),
                       spec_weight=spec_w_t, diff_weight=to_t(diff_w), scatter=to_t(scatter), n_diff=to_t(n_diff))
    out = render_stokes(inp, light, irradiance=irr, mode=mode)
    S = out["S"]                                                       # (H,W,C,3)
    # ---- sensor: exposure, white balance, noise, clipping ----
    S0 = S[..., 0]
    p = float(np.quantile(S0.numpy()[obj], 0.995)) if obj.any() else 1.0
    exposure = float(rng.uniform(*cfg.exposure_range)) / max(p, 1e-3)
    wb = np.exp(rng.normal(0, 0.06, 3)).astype(np.float32) if dr else np.ones(3, np.float32)
    Ssc = S * torch.from_numpy(wb).view(1, 1, 3, 1) * exposure
    th = torch.tensor(list(cfg.angles_deg), dtype=torch.float32) * math.pi / 180
    c, sn = torch.cos(2 * th).view(-1, 1, 1, 1), torch.sin(2 * th).view(-1, 1, 1, 1)
    stack = 0.5 * (Ssc[..., 0] + Ssc[..., 1] * c + Ssc[..., 2] * sn)  # (A,H,W,C)

    def sense(lin: Tensor) -> Tensor:
        sig = torch.sqrt(cfg.read_noise ** 2 + cfg.shot_noise * lin.clamp_min(0))
        out = lin + sig * torch.randn(lin.shape, generator=torch.Generator().manual_seed(int(rng.integers(1 << 31))))
        return out.clamp(0, 1) if cfg.sensor_clip else out

    lin_rgb = sense(Ssc[..., 0])                                      # (H,W,C)
    stack_lin = sense(stack)
    srgb = linear_to_srgb(lin_rgb.permute(2, 0, 1))
    mask = torch.from_numpy(label.astype(np.int64))
    obj_t = torch.from_numpy(obj)
    n_obj = max(int(obj.sum()), 1)
    sev = float(np.isin(label, CORRODED).sum() / n_obj)
    prior = torch.zeros(3, H, W)
    prior[2] = 1.0
    if obj.sum() > 50:                                                  # cartridge-geometry prior from the silhouette only
        try:
            fit = fit_cylinder_from_mask(obj)
            prior, _ = cylinder_normals(H, W, fit["alpha"], fit["cx"], fit["cy_up"], fit["radius"], 0.0)
            prior = torch.where(obj_t.unsqueeze(0), prior, torch.tensor([0, 0, 1.0]).view(3, 1, 1))
        except ValueError:
            pass
    sc = torch.from_numpy(wb).view(1, 1, 3) * exposure
    latent_gt = {   # true latent optical state in the same radiometric scale as lin / S_gt (used for model validation & Fig. 6)
        "D": (out["S_diffuse"][..., 0] * sc).permute(2, 0, 1).contiguous(), "S": (out["S_specular"][..., 0] * sc).permute(2, 0, 1).contiguous(),
        "eta": inp.eta.permute(2, 0, 1).contiguous(), "kappa": inp.kappa.permute(2, 0, 1).contiguous(),
        "n_diff": inp.n_diff.clone(), "rough": rough_t.clone(), "mode": mode,
    }
    return {
        "latent_gt": latent_gt, "rgb": srgb, "lin": lin_rgb.permute(2, 0, 1), "stack": stack_lin.permute(0, 3, 1, 2),          # (A,3,H,W)
        "angles": torch.tensor(list(cfg.angles_deg)), "S_gt": Ssc.permute(2, 3, 0, 1).contiguous(),   # (3colour,3stokes,H,W)
        "normals": n_true.clone() if False else torch.from_numpy(n.astype(np.float32)).permute(2, 0, 1),
        "normals_prior": prior, "mask": mask, "pit_mask": (mask == 6).long(), "obj_mask": obj_t,
        "severity": torch.tensor(sev, dtype=torch.float32),
        "severity_level": torch.bucketize(torch.tensor(sev), torch.tensor(DEFAULT_BINS)),
        "glare_mask": (srgb.max(0).values >= 0.97), "rough_gt": rough_t,
        "meta": dict(group_seed=group["seed"], view_seed=seed, mode=mode, substrate=group["substrate"], level=group["level"],
                     pitted=group["pitted"], exposure=exposure, alpha=alpha),
    }


def generate_dataset(n_groups: int, views_per_group: int, cfg: SynthConfig | None = None, seed: int = 0) -> List[Dict]:
    cfg = cfg or SynthConfig()
    data = []
    for g in range(n_groups):
        grp = make_group(seed * 100003 + g, cfg)
        for v in range(views_per_group):
            d = render_view(grp, seed * 100003 + g * 997 + v + 1, cfg)
            d["group_id"] = g
            data.append(d)
    return data
