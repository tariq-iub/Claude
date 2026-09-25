"""
CPEN: Corrosion Prototype Evolution Network
=============================================
Core primitive: a per-class *ordered trajectory* of learnable prototypes in
embedding space, evolved online via exponential-moving-average vector
quantization (à la VQ-VAE codebooks) but additionally regularised so that
consecutive prototypes within a class trace a smooth 1-manifold ("oxidation
trajectory": healthy -> early oxide -> mature oxide) rather than an
unordered bag of cluster centers. Classification is a soft nearest-prototype
read-out, not a learned linear classifier head.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, rgb_to_hsv
from common.dataset import NUM_CLASSES


class PixelEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, dilation=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, embed_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PrototypeField(nn.Module):
    """
    Prototype bank of shape (num_classes, K, D). Soft assignment probability
    of embedding e to class c:

        p(c | e) = softmax_c( -min_k || e - P_{c,k} ||^2 / tau )

    Trajectory-smoothness regulariser (returned separately for the loss):

        L_traj = sum_c sum_k || P_{c,k+1} - P_{c,k} ||^2

    Online EMA prototype update (VQ-style) with decay `gamma`:

        N_{c,k} <- gamma * N_{c,k} + (1-gamma) * n_{c,k}
        m_{c,k} <- gamma * m_{c,k} + (1-gamma) * sum_{i: assigned to (c,k)} e_i
        P_{c,k} <- m_{c,k} / max(N_{c,k}, eps)
    """

    def __init__(self, num_classes: int, k_per_class: int = 3, dim: int = 32, gamma: float = 0.98, tau: float = 1.0):
        super().__init__()
        self.num_classes, self.k, self.dim = num_classes, k_per_class, dim
        self.gamma, self.tau = gamma, tau
        proto = torch.randn(num_classes, k_per_class, dim) * 0.1
        self.register_buffer("prototypes", proto)
        self.register_buffer("ema_count", torch.ones(num_classes, k_per_class) * 1e-3)
        self.register_buffer("ema_sum", proto.clone())

    def forward(self, emb: torch.Tensor):
        # emb: (B, D, H, W)
        b, d, h, w = emb.shape
        flat = emb.permute(0, 2, 3, 1).reshape(-1, d)  # (N, D)
        # clone so the EMA in-place update below doesn't corrupt the autograd
        # version counter of the tensor `cdist` saved for its backward pass
        proto = self.prototypes.detach().clone().reshape(self.num_classes * self.k, d)  # (C*K, D)
        dist2 = torch.cdist(flat, proto) ** 2  # (N, C*K)
        dist2 = dist2.reshape(-1, self.num_classes, self.k)
        min_dist2, best_k = dist2.min(dim=2)  # (N, C)
        logits = (-min_dist2 / max(self.tau, 1e-4)).reshape(b, h, w, self.num_classes).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                flat_idx = dist2.reshape(-1, self.num_classes * self.k).argmin(dim=1)
                onehot = F.one_hot(flat_idx, self.num_classes * self.k).float()
                count = onehot.sum(0).reshape(self.num_classes, self.k)
                summed = (onehot.t() @ flat).reshape(self.num_classes, self.k, d)
                self.ema_count.mul_(self.gamma).add_(count, alpha=1 - self.gamma)
                self.ema_sum.mul_(self.gamma).add_(summed, alpha=1 - self.gamma)
                self.prototypes.copy_(self.ema_sum / self.ema_count.clamp(min=1e-3).unsqueeze(-1))

        return logits

    def trajectory_loss(self):
        if self.k < 2:
            return torch.tensor(0.0, device=self.prototypes.device)
        diffs = self.prototypes[:, 1:] - self.prototypes[:, :-1]
        return (diffs ** 2).sum(dim=-1).mean()


class CPEN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 32, k_per_class: int = 3):
        super().__init__()
        self.encoder = PixelEncoder(embed_dim)
        self.protos = PrototypeField(num_classes, k_per_class, embed_dim)
        self.last_traj_loss = torch.tensor(0.0)

    def forward(self, rgb: torch.Tensor):
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)
        x = torch.cat([rgb, lab, hsv], dim=1)
        emb = self.encoder(x)
        logits = self.protos(emb)
        self.last_traj_loss = self.protos.trajectory_loss()
        return logits


def build_model():
    return CPEN()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()), "traj_loss=", m.last_traj_loss.item())
