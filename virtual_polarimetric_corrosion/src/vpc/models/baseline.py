"""Baselines: classical image operators, simple pseudo-polarizer (strawman), physics-only virtual polarizer (no
learning), CIELAB/HSV thresholding, RGB-only learned segmenters. DeepLabV3+ / YOLO adapters are intentionally not
bundled (external dependencies); see docs/EXPERIMENTAL_PROTOCOL.md for the adapter contract."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..color import ciede2000, luminance, rgb_to_hsv, rgb_to_lab, srgb_to_linear
from ..inverse.latent_optics import decode_latent
from ..inverse.roughness import roughness_proxy
from ..inverse.specular_diffuse import specular_free_decomposition, specular_probability
from ..optics.fresnel import preset
from ..polarization.virtual_analyzer import VirtualAnalyzer
from .blocks import ConvBNAct
from .vp_corrosion_net import NetConfig, VPCorrosionNet

Tensor = torch.Tensor


# ---------------- classical image operators (numpy, HxWx3 float in [0,1]) ----------------
def apply_clahe(img: np.ndarray, clip: float = 0.01) -> np.ndarray:
    from skimage import exposure, color
    lab = color.rgb2lab(img)
    lab[..., 0] = exposure.equalize_adapthist(lab[..., 0] / 100.0, clip_limit=clip) * 100.0
    return np.clip(color.lab2rgb(lab), 0, 1)


def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    return np.clip(img, 0, 1) ** gamma


def retinex_msr(img: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    x = np.clip(img, 1e-3, 1)
    out = np.zeros_like(x)
    for s in sigmas:
        out += np.log(x) - np.log(np.stack([gaussian_filter(x[..., c], s) for c in range(3)], -1) + 1e-3)
    out /= len(sigmas)
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def highlight_suppression(img: np.ndarray, thresh: float = 0.9, strength: float = 0.8) -> np.ndarray:
    """Clip-and-compress highlights (naive). Operates on value channel; no physics."""
    x = img.copy()
    v = x.max(-1, keepdims=True)
    over = np.maximum(v - thresh, 0)
    return np.clip(x - strength * over, 0, 1)


def specular_removal(img: np.ndarray, specular_color=(1, 1, 1)) -> np.ndarray:
    lin = srgb_to_linear(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float())
    d = specular_free_decomposition(lin, specular_color)["diffuse"]
    from ..color import linear_to_srgb
    return linear_to_srgb(d)[0].permute(1, 2, 0).numpy()


def pseudo_polarizer(img: np.ndarray, theta_deg: float = 0.0, kappa: float = 0.5, thresh: float = 0.8) -> np.ndarray:
    """STRAWMAN: a brightness filter that 'looks polarized': attenuates bright pixels, ignoring Stokes/Fresnel/geometry.
    theta only modulates a global gain, so it has no analyzer physics."""
    gain = 0.75 + 0.25 * np.cos(np.radians(2 * theta_deg))
    v = img.max(-1, keepdims=True)
    return np.clip(img * gain - kappa * np.maximum(v - thresh, 0), 0, 1)


def physics_only_virtual_polarizer(rgb: Tensor, normals: Tensor, theta_deg: float, base: str = "copper",
                                   specular_color=None) -> dict:
    """Physics-only baseline (no learning): specular-free decomposition (specular colour defaults to the copper
    Fresnel tint at normal incidence, NOT white) -> bounded heuristics for roughness -> Fresnel Stokes -> analyzer."""
    lin = srgb_to_linear(rgb)
    n0, k0 = preset(base)
    if specular_color is None:
        from ..optics.fresnel import fresnel_reflectance
        Rs, Rp, _ = fresnel_reflectance(torch.zeros(3), n0, k0)
        specular_color = tuple((0.5 * (Rs + Rp)).tolist())
    dec = specular_free_decomposition(lin, specular_color)
    from ..optics.reflection import diffuse_dolp, roughness_depolarization
    from ..optics.fresnel import fresnel_reflectance
    sprob = specular_probability(lin)
    r = roughness_proxy(luminance(lin), sprob)
    zen = torch.acos(normals[:, 2:3].clamp(-1, 1))
    az = torch.atan2(normals[:, 1:2], normals[:, 0:1])
    Rs, Rp, _ = fresnel_reflectance(zen, n0.view(1, 3, 1, 1), k0.view(1, 3, 1, 1))
    rho_s = (Rs - Rp) / (Rs + Rp) * roughness_depolarization(r)
    rho_d = diffuse_dolp(zen, 1.8)
    S, D = dec["specular"], dec["diffuse"]
    s1 = S * rho_s * torch.cos(2 * (az + np.pi / 2)) + D * rho_d * torch.cos(2 * az)
    s2 = S * rho_s * torch.sin(2 * (az + np.pi / 2)) + D * rho_d * torch.sin(2 * az)
    Sk = torch.stack([S + D, s1, s2], 2)
    return {"S_hat": Sk, "I_theta": VirtualAnalyzer()(Sk, theta_deg)[:, 0]}


# ---------------- colour-space thresholding segmentation ----------------
def lab_threshold_segmentation(img: np.ndarray, healthy_ref_lab: np.ndarray | None = None, thr: float | None = None) -> np.ndarray:
    """Binary corroded mask: CIEDE2000 distance to a healthy-metal reference colour (median Lab of the brightest-saturation
    cluster if not given), Otsu-thresholded unless ``thr`` is provided."""
    from skimage.filters import threshold_otsu
    lab = rgb_to_lab(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float())[0].permute(1, 2, 0).numpy()
    if healthy_ref_lab is None:
        healthy_ref_lab = np.median(lab.reshape(-1, 3), 0)
    de = ciede2000(lab, healthy_ref_lab[None, None])
    t = threshold_otsu(de) if thr is None else thr
    return de > t


def hsv_threshold_segmentation(img: np.ndarray, hue_range=(0.17, 0.55), min_sat: float = 0.15) -> np.ndarray:
    """Greenish (patina-like) hue band in HSV. Heuristic baseline; hue range is a config choice."""
    hsv = rgb_to_hsv(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float())[0].permute(1, 2, 0).numpy()
    return (hsv[..., 0] > hue_range[0]) & (hsv[..., 0] < hue_range[1]) & (hsv[..., 1] > min_sat)


# ---------------- learned RGB-only baselines ----------------
class UNetLite(nn.Module):
    """Compact 3-level U-Net (RGB-only, no physics). Returns same dict keys as VPCorrosionNet for the shared trainer."""

    def __init__(self, in_ch=3, num_classes=7, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        blk = lambda i, o: nn.Sequential(ConvBNAct(i, o), ConvBNAct(o, o))
        self.e1, self.e2, self.e3, self.e4 = blk(in_ch, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3])
        self.d3, self.d2, self.d1 = blk(c[3] + c[2], c[2]), blk(c[2] + c[1], c[1]), blk(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)
        self.pit = nn.Conv2d(c[0], 1, 1)

    def forward(self, rgb, normals_prior=None):
        e1 = self.e1(rgb); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda x, ref: F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(e4, e3), e3], 1)); d2 = self.d2(torch.cat([up(d3, e2), e2], 1)); d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return {"seg_logits": self.head(d1), "pit_logit": self.pit(d1), "feat": d1}


def rgb_only_mobile(num_classes=7, color_spaces=("rgb",), fusion="early") -> VPCorrosionNet:
    """Same backbone/decoder/heads as VP-CorrosionNet with every physics feature switched off ('RGB only' ablation arm)."""
    cfg = NetConfig(num_classes=num_classes, color_spaces=color_spaces, texture=False, specdiff=False, vstokes=False, analyzer=False,
                    geometry=False, roughness=False, psrf=False, uncertainty=False, fusion=fusion)
    return VPCorrosionNet(cfg)


def build_baseline(name: str, **kw) -> nn.Module:
    if name == "unet_lite":
        return UNetLite(**kw)
    if name == "mobile_rgb":
        return rgb_only_mobile(**kw)
    if name in ("deeplabv3plus", "yolo"):
        raise NotImplementedError(f"{name}: external dependency (torchvision / ultralytics). Implement an adapter returning "
                                  "{'seg_logits': (B,K,H,W)} / boxes; see docs/EXPERIMENTAL_PROTOCOL.md")
    raise KeyError(name)
