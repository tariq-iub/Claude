"""
PMD-Net: Perceptual Manifold Diffusion Network
================================================
Core primitive: an *anisotropic reaction-diffusion PDE* integrated on the
pixel lattice, whose diffusion kernel bandwidth is itself learned separately
in the spatial and CIELAB-perceptual domains, and whose closed-form bilateral
kernel (not a softmax content-attention kernel) is applied for a fixed,
learned number of Euler steps. See ../../architectures/01_pmd_net/README.md
for the full theoretical treatment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from common.colorspace import rgb_to_lab, rgb_to_hsv, spatial_coords
from common.dataset import NUM_CLASSES


class LocalPerceptualKernel(nn.Module):
    """
    For every pixel i, computes anisotropic diffusion weights w_ij over a
    (2r+1)x(2r+1) local window using:

        w_ij = exp( - d_s(i,j)^2 / (2 sigma_s^2) - d_c(i,j)^2 / (2 sigma_c^2) )

    where d_s is Euclidean pixel distance and d_c is a *learned, smooth
    surrogate* of CIEDE2000 (a per-channel weighted Euclidean form in a
    learned Lab-affine embedding) so that the whole kernel stays cheap
    (implemented via unfold) yet perceptually calibrated. sigma_s, sigma_c
    are learned scalars (log-parametrized for positivity).
    """

    def __init__(self, radius: int = 3):
        super().__init__()
        self.radius = radius
        k = 2 * radius + 1
        self.k = k
        # learned affine map calibrating Euclidean Lab distance towards ΔE00
        self.lab_affine = nn.Parameter(torch.eye(3) * torch.tensor([1.0, 1.4, 1.4]))
        self.log_sigma_s = nn.Parameter(torch.tensor(0.5).log())
        self.log_sigma_c = nn.Parameter(torch.tensor(6.0).log())

        offs = torch.arange(-radius, radius + 1, dtype=torch.float32)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        self.register_buffer("d_s2", (ox ** 2 + oy ** 2).reshape(1, 1, k * k, 1, 1))

    def forward(self, lab: torch.Tensor):
        b, c, h, w = lab.shape
        k, r = self.k, self.radius
        lab_t = torch.einsum("bchw,oc->bohw", lab, self.lab_affine)
        patches = F.unfold(lab_t, kernel_size=k, padding=r)  # (B, 3*k*k, H*W)
        patches = patches.reshape(b, 3, k * k, h, w)
        center = lab_t.unsqueeze(2)  # (B,3,1,H,W)
        d_c2 = ((patches - center) ** 2).sum(dim=1, keepdim=True)  # (B,1,k*k,H,W)
        sigma_s2 = (self.log_sigma_s.exp() ** 2).clamp(min=1e-3)
        sigma_c2 = (self.log_sigma_c.exp() ** 2).clamp(min=1e-3)
        logits = -self.d_s2 / (2 * sigma_s2) - d_c2 / (2 * sigma_c2)
        weights = F.softmax(logits, dim=2)  # normalize over neighborhood -> (B,1,k*k,H,W)
        return weights


class DiffusionStep(nn.Module):
    def __init__(self, channels: int, radius: int = 3):
        super().__init__()
        self.radius = radius
        self.k = 2 * radius + 1
        self.eta = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat: torch.Tensor, weights: torch.Tensor):
        b, c, h, w = feat.shape
        k, r = self.k, self.radius
        patches = F.unfold(feat, kernel_size=k, padding=r).reshape(b, c, k * k, h, w)
        agg = (patches * weights).sum(dim=2)  # weighted neighborhood average
        return feat + torch.sigmoid(self.eta) * (agg - feat)


class PMDNet(nn.Module):
    """Perceptual Manifold Diffusion Network."""

    def __init__(self, num_classes: int = NUM_CLASSES, feat_dim: int = 24, radius: int = 3, steps: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(8, feat_dim, 3, padding=1), nn.GroupNorm(4, feat_dim), nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.GroupNorm(4, feat_dim), nn.GELU(),
        )
        self.kernel = LocalPerceptualKernel(radius=radius)
        self.diffusion_steps = nn.ModuleList([DiffusionStep(feat_dim, radius) for _ in range(steps)])
        self.head = nn.Conv2d(feat_dim, num_classes, 1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)
        stack = torch.cat([rgb, lab, hsv[:, :2]], dim=1)  # rgb(3)+lab(3)+hue,sat(2) = 8 channels
        feat = self.stem(stack)
        weights = self.kernel(lab)
        for step in self.diffusion_steps:
            feat = step(feat, weights)
        return self.head(feat)


def build_model():
    return PMDNet()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()))
