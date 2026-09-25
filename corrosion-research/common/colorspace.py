"""
Differentiable colour-space utilities shared by every architecture.

All conversions operate on PyTorch tensors so that colour-space reasoning
can sit *inside* the computational graph (gradients flow back to RGB / to
whatever produced the RGB image) rather than being a fixed, non-differentiable
pre-processing step.

Conventions
-----------
RGB   : float tensor in [0, 1], shape (..., 3, H, W) or (..., 3)
Lab   : CIE 1976 L*a*b*, D65 white point, L in [0, 100], a,b roughly [-128,127]
HSV   : H in [0, 1] (fraction of 360 deg), S, V in [0, 1]

References
----------
- Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
  Implementation Notes, Supplementary Test Data, and Mathematical Observations".
- Standard sRGB -> CIEXYZ -> CIELAB pipeline (IEC 61966-2-1).
"""
from __future__ import annotations

import math
import torch


# --------------------------------------------------------------------------- #
# sRGB -> Lab
# --------------------------------------------------------------------------- #

_XYZ_FROM_LINEAR_RGB = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)

_WHITE_D65 = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)).clamp(min=1e-8) ** 2.4
    )


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(
        t > delta ** 3, t.clamp(min=1e-8) ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0
    )


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W) in [0,1] -> lab: (B,3,H,W)."""
    assert rgb.shape[-3] == 3
    lin = _srgb_to_linear(rgb)
    mat = _XYZ_FROM_LINEAR_RGB.to(rgb.device, rgb.dtype)
    # (B,3,H,W) -> matmul over channel dim
    b, c, h, w = lin.shape
    flat = lin.reshape(b, c, h * w)
    xyz = torch.einsum("oc,bcn->bon", mat, flat).reshape(b, 3, h, w)
    white = _WHITE_D65.to(rgb.device, rgb.dtype).view(1, 3, 1, 1)
    xyz_n = xyz / white
    fx, fy, fz = _f_lab(xyz_n[:, 0]), _f_lab(xyz_n[:, 1]), _f_lab(xyz_n[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return torch.stack([L, a, b_], dim=1)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W) in [0,1] -> hsv: (B,3,H,W), all channels in [0,1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    v = maxc
    delta = (maxc - minc).clamp(min=1e-8)
    s = torch.where(maxc > 1e-8, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))

    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta

    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta <= 1e-8, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=1)


def chroma_intensity(lab: torch.Tensor) -> torch.Tensor:
    """Chroma C* = sqrt(a*^2+b*^2) and intensity (here: L*) stacked -> (B,2,H,W)."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    c = torch.sqrt((a ** 2 + b ** 2).clamp(min=1e-12))
    return torch.stack([c, L], dim=1)


def spatial_coords(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalised (x,y) coordinate grid -> (2,H,W), range [-1,1]."""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


# --------------------------------------------------------------------------- #
# CIEDE2000
# --------------------------------------------------------------------------- #

def deltaE2000(lab1: torch.Tensor, lab2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Fully differentiable CIEDE2000 colour difference.

    lab1, lab2 : (..., 3) tensors (broadcastable), channel order L*,a*,b*.
    returns    : (...,) tensor of Delta E00 values.

    This is the *exact* CIEDE2000 formula (not a Euclidean approximation),
    implemented with smooth, differentiable primitives. atan2 and sqrt are
    differentiable everywhere except exactly at the origin, which we guard
    with `eps`.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = torch.sqrt((a1 ** 2 + b1 ** 2).clamp(min=eps))
    C2 = torch.sqrt((a2 ** 2 + b2 ** 2).clamp(min=eps))
    Cbar = (C1 + C2) / 2.0

    G = 0.5 * (1 - torch.sqrt((Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)).clamp(min=eps)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = torch.sqrt((a1p ** 2 + b1 ** 2).clamp(min=eps))
    C2p = torch.sqrt((a2p ** 2 + b2 ** 2).clamp(min=eps))

    h1p = torch.atan2(b1, a1p + eps) % (2 * math.pi)
    h2p = torch.atan2(b2, a2p + eps) % (2 * math.pi)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > math.pi, dhp - 2 * math.pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + 2 * math.pi, dhp)
    dhp = torch.where((C1p * C2p) < eps, torch.zeros_like(dhp), dhp)
    dHp = 2 * torch.sqrt((C1p * C2p).clamp(min=0.0)) * torch.sin(dhp / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    hsum = h1p + h2p
    habs = torch.abs(h1p - h2p)
    Hbarp = torch.where(
        (C1p * C2p) < eps,
        hsum,
        torch.where(
            habs <= math.pi,
            hsum / 2.0,
            torch.where(hsum < 2 * math.pi, (hsum + 2 * math.pi) / 2.0, (hsum - 2 * math.pi) / 2.0),
        ),
    )

    T = (
        1
        - 0.17 * torch.cos(Hbarp - math.radians(30))
        + 0.24 * torch.cos(2 * Hbarp)
        + 0.32 * torch.cos(3 * Hbarp + math.radians(6))
        - 0.20 * torch.cos(4 * Hbarp - math.radians(63))
    )

    Hbarp_deg = Hbarp * 180.0 / math.pi
    d_theta = math.radians(30) * torch.exp(-(((Hbarp_deg - 275) / 25) ** 2))
    Rc = 2 * torch.sqrt((Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7)).clamp(min=eps))
    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / torch.sqrt((20 + (Lbarp - 50) ** 2).clamp(min=eps))
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -torch.sin(2 * d_theta) * Rc

    kL = kC = kH = 1.0
    dE = torch.sqrt(
        (
            (dLp / (kL * Sl)) ** 2
            + (dCp / (kC * Sc)) ** 2
            + (dHp / (kH * Sh)) ** 2
            + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
        ).clamp(min=0.0)
        + eps
    )
    return dE


def spatial_distance(xy1: torch.Tensor, xy2: torch.Tensor) -> torch.Tensor:
    """d_s(i,j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2). xy*: (...,2)."""
    return torch.sqrt(((xy1 - xy2) ** 2).sum(dim=-1).clamp(min=1e-12))


def build_perceptual_stack(rgb: torch.Tensor) -> torch.Tensor:
    """
    Concatenate RGB, Lab, HSV, chroma/intensity and spatial coordinates into a
    single (B, 3+3+3+2+2=13, H, W) tensor used as the canonical input
    representation across architectures (each architecture then decides which
    subset of channels it consumes).
    """
    b, _, h, w = rgb.shape
    lab = rgb_to_lab(rgb)
    hsv = rgb_to_hsv(rgb)
    ci = chroma_intensity(lab)
    xy = spatial_coords(h, w, device=rgb.device, dtype=rgb.dtype).unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat([rgb, lab, hsv, ci, xy], dim=1)
