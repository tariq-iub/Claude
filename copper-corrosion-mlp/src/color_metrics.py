"""sRGB <-> CIELAB <-> HSV conversion and CIEDE2000 perceptual distance.

Pure NumPy, vectorized, no image-processing dependency. D65 white point,
IEC 61966-2-1 sRGB companding. CIEDE2000 follows Sharma, Wu & Dalal (2005).
"""
from __future__ import annotations

import numpy as np

_XYZ_FROM_LINEAR_RGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
_DELTA = 6.0 / 29.0


def srgb_to_lab(rgb_255: np.ndarray) -> np.ndarray:
    """rgb_255: (..., 3) in [0,255] -> Lab (..., 3)."""
    rgb = np.asarray(rgb_255, dtype=np.float64) / 255.0
    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)
    xyz = lin @ _XYZ_FROM_LINEAR_RGB.T
    xyz_n = xyz / _WHITE_D65
    f = np.where(xyz_n > _DELTA**3, np.cbrt(np.clip(xyz_n, 1e-12, None)),
                 xyz_n / (3 * _DELTA**2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a_ = 500.0 * (f[..., 0] - f[..., 1])
    b_ = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a_, b_], axis=-1)


def rgb_to_hsv(rgb_255: np.ndarray) -> np.ndarray:
    """rgb_255: (..., 3) in [0,255] -> HSV (..., 3), H in [0,1) fraction of 360deg, S,V in [0,1]."""
    rgb = np.asarray(rgb_255, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 1e-12, delta / np.clip(maxc, 1e-12, None), 0.0)

    rc = np.where(delta > 1e-12, (maxc - r) / np.clip(delta, 1e-12, None), 0.0)
    gc = np.where(delta > 1e-12, (maxc - g) / np.clip(delta, 1e-12, None), 0.0)
    bc = np.where(delta > 1e-12, (maxc - b) / np.clip(delta, 1e-12, None), 0.0)

    h = np.zeros_like(v)
    is_r = (maxc == r) & (delta > 1e-12)
    is_g = (maxc == g) & (delta > 1e-12)
    is_b = (maxc == b) & (delta > 1e-12)
    h = np.where(is_r, bc - gc, h)
    h = np.where(is_g & ~is_r, 2.0 + rc - bc, h)
    h = np.where(is_b & ~is_r & ~is_g, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def lab_chroma(lab: np.ndarray) -> np.ndarray:
    """C* = sqrt(a*^2 + b*^2)."""
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Pairwise CIEDE2000 distance.

    lab1: (N,3), lab2: (M,3) -> (N,M). Broadcasts if either is (3,).
    """
    lab1 = np.atleast_2d(lab1).astype(np.float64)
    lab2 = np.atleast_2d(lab2).astype(np.float64)
    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[None, :, 0], lab2[None, :, 1], lab2[None, :, 2]

    kL = kC = kH = 1.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(np.abs(dhp) > 180, dhp - 360 * np.sign(dhp), dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    hsum = h1p + h2p
    habsdiff = np.abs(h1p - h2p)
    hbarp = np.where(
        (C1p * C2p) == 0, hsum,
        np.where(habsdiff <= 180, hsum / 2,
                 np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)),
    )
    T = (1 - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    dtheta = 30 * np.exp(-(((hbarp - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7))
    SL = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC

    dE = np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE
