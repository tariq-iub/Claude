"""Differentiable colour-space utilities (sRGB, linear RGB, HSV, CIELAB) and CIEDE2000.

All torch functions operate on tensors shaped (..., 3, H, W) or (B, 3, H, W) with channel
dimension ``dim=-3`` and RGB values in [0, 1]. CIELAB uses the D65 white point.
CIEDE2000 (numpy) follows Sharma, Wu & Dalal (2005).
"""
from __future__ import annotations

import math

import numpy as np
import torch

_EPS = 1e-8

_RGB2XYZ = torch.tensor(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)
_WHITE_D65 = torch.tensor([0.95047, 1.0, 1.08883])
LUMA_WEIGHTS = (0.2126729, 0.7151522, 0.0721750)  # Rec.709 luminance of linear RGB


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(_EPS) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(_EPS) ** (1 / 2.4) - 0.055)


def luminance(lin: torch.Tensor) -> torch.Tensor:
    """Linear-RGB luminance, keeps a singleton channel dim: (..., 1, H, W)."""
    w = torch.tensor(LUMA_WEIGHTS, dtype=lin.dtype, device=lin.device).view(-1, 1, 1)
    return (lin * w).sum(dim=-3, keepdim=True)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """H in [0,1) (hue/360), S, V in [0,1]. Hue is circular: use sin/cos for learning (see hsv_features)."""
    r, g, b = rgb.unbind(dim=-3)
    maxc, _ = rgb.max(dim=-3)
    minc, _ = rgb.min(dim=-3)
    v = maxc
    d = maxc - minc
    s = d / (maxc + _EPS)
    dd = d + _EPS
    rc, gc, bc = (maxc - r) / dd, (maxc - g) / dd, (maxc - b) / dd
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(d < 1e-6, torch.zeros_like(h), (h / 6.0) % 1.0)
    return torch.stack([h, s, v], dim=-3)


def hsv_features(rgb: torch.Tensor) -> torch.Tensor:
    """Circular-safe HSV feature stack: [cos(2*pi*H), sin(2*pi*H), S, V] (4 channels)."""
    hsv = rgb_to_hsv(rgb)
    h = hsv[..., 0, :, :] * 2 * math.pi
    return torch.stack([torch.cos(h), torch.sin(h), hsv[..., 1, :, :], hsv[..., 2, :, :]], dim=-3)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB (gamma-encoded, [0,1]) -> CIELAB (D65). Returns (L*, a*, b*) with L in [0,100]."""
    lin = srgb_to_linear(rgb)
    m = _RGB2XYZ.to(lin)
    xyz = torch.einsum("ij,...jhw->...ihw", m, lin)
    wp = _WHITE_D65.to(lin).view(3, 1, 1)
    t = xyz / wp
    delta = 6.0 / 29.0
    f = torch.where(t > delta ** 3, t.clamp_min(_EPS) ** (1 / 3), t / (3 * delta ** 2) + 4.0 / 29.0)
    fx, fy, fz = f.unbind(dim=-3)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-3)


def lab_normalized(rgb: torch.Tensor) -> torch.Tensor:
    """Lab scaled to roughly [0,1] / [-1,1] for network input: (L/100, a/128, b/128)."""
    lab = rgb_to_lab(rgb)
    scale = torch.tensor([100.0, 128.0, 128.0], dtype=lab.dtype, device=lab.device).view(3, 1, 1)
    return lab / scale


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (numpy, broadcastable (...,3)). Auxiliary perceptual feature only;
    it is NOT evidence of chemical composition."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp, Cbp = 0.5 * (L1 + L2), 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = np.where(C1p * C2p == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


COLOR_SPACES = ("rgb", "lab", "hsv")


def color_features(rgb: torch.Tensor, spaces=("rgb",)) -> torch.Tensor:
    """Concatenate colour representations (channel dim -3). ``spaces`` subset of COLOR_SPACES.
    rgb: 3 ch; lab: 3 ch (normalised); hsv: 4 ch (cos H, sin H, S, V)."""
    out = []
    for s in spaces:
        if s == "rgb":
            out.append(rgb)
        elif s == "lab":
            out.append(lab_normalized(rgb))
        elif s == "hsv":
            out.append(hsv_features(rgb))
        else:
            raise ValueError(f"unknown colour space {s!r}; choose from {COLOR_SPACES}")
    return torch.cat(out, dim=-3)


def color_channels(spaces) -> int:
    return sum({"rgb": 3, "lab": 3, "hsv": 4}[s] for s in spaces)
