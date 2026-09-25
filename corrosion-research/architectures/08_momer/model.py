"""
MoMER: Mixture-of-Material-Experts Router Network
====================================================
Core primitive: a bank of tiny experts, each specialised (by training
dynamics) to a distinct corrosion *appearance regime*, is routed not by a
purely learned black-box gate but by a gate that is the product of (a) a
learned content logit and (b) an explicit *colour prior* -- the likelihood
of the local patch's mean Lab under a small bank of learnable per-class
Gaussian reference centroids:

    g_e = softmax_e( z_e + log N(Lab_patch; mu_e, Sigma_e) )

The entropy of the resulting routing distribution H(g) is exposed as a
genuine *per-pixel epistemic-uncertainty map* (ambiguous / boundary /
out-of-distribution appearance regions naturally produce high routing
entropy), rather than uncertainty being bolted on via MC-dropout or an
auxiliary head disconnected from the computation that produced the
prediction.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, rgb_to_hsv
from common.dataset import NUM_CLASSES


class TinyExpert(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, width, 3, padding=1), nn.GroupNorm(2, width), nn.GELU(),
            nn.Conv2d(width, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class ColorPriorRouter(nn.Module):
    def __init__(self, in_dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.gate_conv = nn.Conv2d(in_dim, n_experts, 1)
        self.mu = nn.Parameter(torch.randn(n_experts, 3) * 20.0)
        self.log_sigma = nn.Parameter(torch.zeros(n_experts, 3))

    def forward(self, feat: torch.Tensor, lab_local_mean: torch.Tensor):
        # lab_local_mean: (B,3,H,W) - locally smoothed Lab (patch statistic proxy)
        content_logits = self.gate_conv(feat)  # (B,E,H,W)
        b, _, h, w = lab_local_mean.shape
        lab_flat = lab_local_mean.permute(0, 2, 3, 1).unsqueeze(3)  # (B,H,W,1,3)
        mu = self.mu.view(1, 1, 1, self.n_experts, 3)
        sigma2 = (self.log_sigma.exp() ** 2).clamp(min=1.0).view(1, 1, 1, self.n_experts, 3)
        log_prior = -0.5 * (((lab_flat - mu) ** 2) / sigma2).sum(-1) - 0.5 * torch.log(sigma2.prod(-1))
        log_prior = log_prior.permute(0, 3, 1, 2)  # (B,E,H,W)
        gate_logits = content_logits + log_prior
        gate = F.softmax(gate_logits, dim=1)
        return gate


def local_mean(x: torch.Tensor, k: int = 5):
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)


class MoMER(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, n_experts: int = 5, feat_dim: int = 20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(9, feat_dim, 3, padding=1), nn.GroupNorm(4, feat_dim), nn.GELU(),
        )
        self.experts = nn.ModuleList([TinyExpert(feat_dim, num_classes) for _ in range(n_experts)])
        self.router = ColorPriorRouter(feat_dim, n_experts)
        self.n_experts = n_experts
        self.last_uncertainty = None

    def forward(self, rgb: torch.Tensor):
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)
        x = torch.cat([rgb, lab, hsv], dim=1)
        feat = self.stem(x)
        lab_smoothed = local_mean(lab)

        gate = self.router(feat, lab_smoothed)  # (B,E,H,W)
        expert_outs = torch.stack([e(feat) for e in self.experts], dim=1)  # (B,E,C,H,W)
        logits = (gate.unsqueeze(2) * expert_outs).sum(dim=1)  # (B,C,H,W)

        entropy = -(gate.clamp(min=1e-8) * gate.clamp(min=1e-8).log()).sum(dim=1) / torch.log(
            torch.tensor(float(self.n_experts))
        )
        self.last_uncertainty = entropy  # (B,H,W) in [0,1], per-pixel routing uncertainty
        return logits


def build_model():
    return MoMER()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()))
    print("uncertainty map:", m.last_uncertainty.shape, m.last_uncertainty.mean().item())
