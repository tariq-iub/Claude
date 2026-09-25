"""
INCF: Implicit Neural Corrosion Field
========================================
Core primitive: rather than producing a fixed H x W logit tensor via
transposed convolutions/upsampling, a lightweight CNN encoder produces a
low-resolution latent feature grid, and a coordinate-based implicit
function (sinusoidal/SIREN-style MLP) maps (x, y, bilinearly-sampled local
latent, local Lab) -> class logits *at any continuous coordinate*. The
segmentation "mask" is a continuous field F: R^2 -> R^C that can be queried
at arbitrary (super-)resolution at inference time -- e.g. 4x denser than the
training grid for crisper boundaries -- which a discrete decoder cannot do
without retraining or naive interpolation of already-discretised logits.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, rgb_to_hsv
from common.dataset import NUM_CLASSES


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim: int = 24, downsample: int = 4):
        super().__init__()
        self.downsample = downsample
        layers = []
        in_c = 9
        for _ in range(int(torch.log2(torch.tensor(downsample)).item())):
            layers += [nn.Conv2d(in_c, latent_dim, 3, stride=2, padding=1), nn.GroupNorm(4, latent_dim), nn.GELU()]
            in_c = latent_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, D, H/ds, W/ds)


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, omega0=30.0, is_first=False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = 1 / in_f if is_first else (6 / in_f) ** 0.5 / omega0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class ImplicitField(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        in_dim = 2 + latent_dim + 3  # (x,y) + local latent + local Lab
        layers = [SineLayer(in_dim, hidden, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, coords, local_latent, local_lab):
        inp = torch.cat([coords, local_latent, local_lab], dim=-1)
        return self.out(self.body(inp))


class INCF(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, latent_dim: int = 24, downsample: int = 4):
        super().__init__()
        self.encoder = LatentEncoder(latent_dim, downsample)
        self.field = ImplicitField(latent_dim, num_classes)
        self.latent_dim = latent_dim

    def forward(self, rgb: torch.Tensor, query_hw: tuple | None = None):
        b, _, h, w = rgb.shape
        qh, qw = query_hw if query_hw is not None else (h, w)
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)
        x = torch.cat([rgb, lab, hsv], dim=1)
        latent = self.encoder(x)  # (B,D,h',w')

        ys = torch.linspace(-1, 1, qh, device=rgb.device)
        xs = torch.linspace(-1, 1, qw, device=rgb.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)  # (B,qh,qw,2)

        local_latent = F.grid_sample(latent, grid, mode="bilinear", align_corners=True)  # (B,D,qh,qw)
        local_lab = F.grid_sample(lab, grid, mode="bilinear", align_corners=True)  # (B,3,qh,qw)

        coords = grid.reshape(b, qh * qw, 2)
        local_latent_flat = local_latent.permute(0, 2, 3, 1).reshape(b, qh * qw, self.latent_dim)
        local_lab_flat = local_lab.permute(0, 2, 3, 1).reshape(b, qh * qw, 3)

        logits_flat = self.field(coords, local_latent_flat, local_lab_flat)  # (B, qh*qw, C)
        logits = logits_flat.permute(0, 2, 1).reshape(b, -1, qh, qw)
        return logits


def build_model():
    return INCF()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print("native res:", y.shape)
    y_hires = m(x, query_hw=(128, 128))
    print("super-res query:", y_hires.shape)
    print("params:", sum(p.numel() for p in m.parameters()))
