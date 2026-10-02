"""Perturbations for degradation curves. All operate on sRGB tensors (B,3,H,W) in [0,1]; geometric ones also transform masks."""
from __future__ import annotations

import io
import math
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..color import linear_to_srgb, srgb_to_linear

Tensor = torch.Tensor


def illumination_intensity(x: Tensor, factor: float) -> Tensor:
    return linear_to_srgb(srgb_to_linear(x) * factor)


def exposure_stops(x: Tensor, stops: float) -> Tensor:
    """Scale linear radiance by 2^stops then clip (overexposure produces real clipping/glare)."""
    return linear_to_srgb((srgb_to_linear(x) * 2.0 ** stops).clamp(0, 1))


def color_temperature(x: Tensor, shift: float) -> Tensor:
    """shift > 0 warmer (more red / less blue), < 0 cooler; linear-light channel gains."""
    g = torch.tensor([math.exp(0.35 * shift), 1.0, math.exp(-0.35 * shift)], device=x.device).view(1, 3, 1, 1)
    return linear_to_srgb((srgb_to_linear(x) * g).clamp(0, 1))


def white_balance_error(x: Tensor, gain: float) -> Tensor:
    g = torch.tensor([gain, 1.0, 1.0 / gain], device=x.device).view(1, 3, 1, 1)
    return linear_to_srgb((srgb_to_linear(x) * g).clamp(0, 1))


def add_glare(x: Tensor, strength: float, seed: int = 0) -> Tensor:
    """Adds a saturating elliptical Gaussian glare blob (strength in [0,1+]); unpolarized white additive light."""
    g = torch.Generator().manual_seed(seed)
    B, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    out = []
    for b in range(B):
        cx, cy = (torch.rand(2, generator=g) * 1.2 - 0.6).tolist()
        sx, sy = (0.15 + 0.3 * torch.rand(2, generator=g)).tolist()
        blob = torch.exp(-((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2)))
        out.append(blob)
    blob = torch.stack(out).unsqueeze(1).to(x.device)
    return linear_to_srgb((srgb_to_linear(x) + strength * 2.0 * blob).clamp(0, 1))


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return x
    k = int(2 * math.ceil(3 * sigma) + 1)
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    w = torch.exp(-ax ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    y = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def gaussian_noise(x: Tensor, sigma: float, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return (x + sigma * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1)


def jpeg_compression(x: Tensor, quality: int) -> Tensor:
    from PIL import Image
    out = []
    for im in x.detach().cpu():
        arr = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG", quality=int(quality)); buf.seek(0)
        out.append(torch.from_numpy(np.asarray(Image.open(buf)).astype(np.float32) / 255).permute(2, 0, 1))
    return torch.stack(out).to(x.device)


def rotate(x: Tensor, deg: float, mask: Optional[Tensor] = None):
    """Rotate image (bilinear) and mask (nearest) about the centre; reflection padding for image, background(0) for mask."""
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xr = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if mask is None:
        return xr
    mr = F.grid_sample(mask.unsqueeze(1).float(), grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(1).long()
    return xr, mr


def camera_shift(x: Tensor, dx: int, dy: int, mask: Optional[Tensor] = None):
    xr = torch.roll(x, (dy, dx), (2, 3))
    return xr if mask is None else (xr, torch.roll(mask, (dy, dx), (1, 2)))


def background_replace(x: Tensor, obj_mask: Tensor, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    bg = torch.rand(x.shape[0], 3, 1, 1, generator=g).to(x.device) * 0.8 + 0.1
    noise = 0.05 * torch.randn(x.shape, generator=g).to(x.device)
    return torch.where(obj_mask.unsqueeze(1), x, (bg + noise).clamp(0, 1))


PHOTOMETRIC: Dict[str, tuple] = {
    "illumination": (illumination_intensity, [0.25, 0.5, 1.0, 2.0, 4.0]),
    "exposure_stops": (exposure_stops, [-2, -1, 0, 1, 2]),
    "color_temperature": (color_temperature, [-1.0, -0.5, 0.0, 0.5, 1.0]),
    "white_balance": (white_balance_error, [0.7, 0.85, 1.0, 1.15, 1.3]),
    "glare": (add_glare, [0.0, 0.25, 0.5, 1.0, 2.0]),
    "blur_sigma": (gaussian_blur, [0.0, 0.5, 1.0, 2.0, 3.0]),
    "noise_sigma": (gaussian_noise, [0.0, 0.02, 0.05, 0.1, 0.2]),
    "jpeg_quality": (jpeg_compression, [95, 70, 40, 20, 10]),
}


def degradation_curve(predict_fn: Callable[[Tensor], Tensor], rgb: Tensor, mask: Tensor, name: str, levels=None) -> list[dict]:
    """predict_fn(rgb)->class map (B,H,W). Returns raw rows [{perturbation, level, miou, ...}] (measured points; fit separately)."""
    from .metrics import SegMetrics
    fn, default = PHOTOMETRIC[name]
    rows = []
    for lv in (levels if levels is not None else default):
        pred = predict_fn(fn(rgb, lv))
        m = SegMetrics(int(max(mask.max().item(), pred.max().item())) + 1)
        m.update(pred.cpu().numpy(), mask.cpu().numpy())
        rows.append({"perturbation": name, "level": lv, **m.summary()})
    return rows
