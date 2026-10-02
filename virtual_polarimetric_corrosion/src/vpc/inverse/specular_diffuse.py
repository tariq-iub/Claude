"""Specular-diffuse decomposition from a single RGB image (physics-informed heuristics + learned head).

IMPORTANT physics note: the dichromatic model assumes the specular reflection has the colour of the illuminant. That
is TRUE for dielectrics but FALSE for metals: a conductor's specular reflection is tinted by its Fresnel reflectance
(copper's highlights are reddish at normal incidence). ``specular_color`` therefore must be set per material; the
default white-highlight assumption is only appropriate near grazing angles (R_s,R_p -> 1) and for corrosion layers.
Also, neither component is assumed (un)polarized: polarization of each component is modelled downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def specular_free_decomposition(lin_rgb: Tensor, specular_color=(1.0, 1.0, 1.0), eps: float = 1e-4) -> dict:
    """Shen & Zheng-style min-channel separation in linear RGB, generalised to a non-white specular colour w.
    I = D + s*w with s = min_c(I_c / w_c) offset-removed by the image-wide minimum chroma (camera/ambient offset).
    Returns dict(diffuse, specular, spec_scalar). lin_rgb (B,3,H,W) in [0,1]."""
    w = torch.tensor(specular_color, dtype=lin_rgb.dtype, device=lin_rgb.device).view(1, 3, 1, 1)
    w = w / w.max()
    ratio = lin_rgb / w.clamp_min(eps)
    m = ratio.min(dim=1, keepdim=True).values
    floor = m.flatten(1).quantile(0.05, dim=1).view(-1, 1, 1, 1)  # remove diffuse floor so s>=0 is highlight excess
    s = (m - floor).clamp_min(0)
    spec = s * w
    return {"diffuse": (lin_rgb - spec).clamp_min(0), "specular": spec, "spec_scalar": s}


def specular_probability(lin_rgb: Tensor, sharpness: float = 12.0, v_thresh: float = 0.75) -> Tensor:
    """Soft highlight likelihood from value (brightness) and local low-saturation; (B,1,H,W). Heuristic prior only."""
    v = lin_rgb.max(1, keepdim=True).values
    mn = lin_rgb.min(1, keepdim=True).values
    sat = 1 - mn / (v + 1e-6)
    return torch.sigmoid(sharpness * (v - v_thresh)) * (0.5 + 0.5 * (1 - sat))


def glare_likelihood(srgb: Tensor, sat_level: float = 0.97, blur: int = 5) -> Tensor:
    """Likelihood that a pixel is saturated/clipped glare (B,1,H,W): fraction of near-saturated pixels in a neighbourhood."""
    sat = (srgb.max(1, keepdim=True).values >= sat_level).float()
    return F.avg_pool2d(sat, blur, 1, blur // 2)


class SpecularDiffuseHead(nn.Module):
    """Learned soft decomposition on top of shared features: predicts diffuse RGB (positive) and specular fraction."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, 4, 1))

    def forward(self, feat: Tensor, lin_rgb: Tensor) -> dict:
        o = self.net(feat)
        frac = torch.sigmoid(o[:, :1])  # specular fraction of luminance
        tint = torch.softmax(o[:, 1:4], 1) * 3.0  # specular colour (sums to 3, ~1 each when white)
        spec = lin_rgb * frac * tint / tint.mean(1, keepdim=True).clamp_min(1e-6) * 1.0
        spec = torch.minimum(spec, lin_rgb)
        return {"specular": spec, "diffuse": lin_rgb - spec, "spec_frac": frac}
