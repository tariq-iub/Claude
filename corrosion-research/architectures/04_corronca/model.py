"""
CorroNCA: Corrosion Neural Cellular Automaton
================================================
Core primitive: segmentation is the *emergent steady state* of a local,
stochastic, iterative update rule applied identically at every pixel
("cell"), in the spirit of Mordvintsev et al.'s growing neural cellular
automata, but re-purposed here as a recognition (not generation) engine:
the automaton's hidden state is initialised from perceptual features and
iterated until it settles into a self-consistent per-pixel class read-out.
This gives an extremely small, translation-equivariant, anytime model
(inference can be stopped after any number of steps) fundamentally
different from feed-forward CNN/Transformer encoder-decoders.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, deltaE2000
from common.dataset import NUM_CLASSES

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
_SOBEL_Y = _SOBEL_X.t()
_LAPLACE = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 16.0


class PerceptionFilter(nn.Module):
    """
    Fixed depthwise Sobel/Laplacian perception (as in classic NCA) PLUS a
    perceptual-gradient channel: the ΔE00 distance between each cell and its
    4-neighbours, giving the automaton direct access to *how much the
    material colour is changing* locally -- the corrosion-specific inductive
    bias (material transitions correlate with high local ΔE00).
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        kernels = torch.stack([_SOBEL_X, _SOBEL_Y, _LAPLACE], dim=0).unsqueeze(1)  # (3,1,3,3)
        self.register_buffer("kernels", kernels.repeat(state_dim, 1, 1, 1))  # depthwise

    def forward(self, state: torch.Tensor, lab: torch.Tensor):
        b, c, h, w = state.shape
        grad = F.conv2d(F.pad(state, (1, 1, 1, 1), mode="replicate"), self.kernels, groups=c)
        grad = grad.reshape(b, c, 3, h, w).reshape(b, c * 3, h, w)

        lab_pad = F.pad(lab, (1, 1, 1, 1), mode="replicate")
        center = lab
        left = lab_pad[:, :, 1:-1, 0:-2]
        right = lab_pad[:, :, 1:-1, 2:]
        up = lab_pad[:, :, 0:-2, 1:-1]
        down = lab_pad[:, :, 2:, 1:-1]

        def de(a, b_):
            return deltaE2000(a.permute(0, 2, 3, 1), b_.permute(0, 2, 3, 1)).unsqueeze(1)

        de_map = torch.cat([de(center, left), de(center, right), de(center, up), de(center, down)], dim=1)
        return torch.cat([state, grad, de_map], dim=1)


class CorroNCA(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden_dim: int = 12, steps: int = 12, fire_rate: float = 0.6):
        super().__init__()
        self.num_classes = num_classes
        self.state_dim = num_classes + hidden_dim  # first `num_classes` channels = class logits
        self.steps = steps
        self.fire_rate = fire_rate
        self.perception = PerceptionFilter(self.state_dim)
        perc_dim = self.state_dim * (1 + 3) + 4  # state + 3 filters*state_dim + 4 deltaE channels
        self.update_rule = nn.Sequential(
            nn.Conv2d(perc_dim, 32, 1), nn.GELU(),
            nn.Conv2d(32, self.state_dim, 1),
        )
        nn.init.zeros_(self.update_rule[-1].weight)
        nn.init.zeros_(self.update_rule[-1].bias)
        self.stem = nn.Conv2d(6, self.state_dim, 1)

    def forward(self, rgb: torch.Tensor):
        lab = rgb_to_lab(rgb)
        b, _, h, w = rgb.shape
        state = self.stem(torch.cat([rgb, lab], dim=1))

        for _ in range(self.steps):
            perc = self.perception(state, lab)
            delta = self.update_rule(perc)
            if self.training:
                fire_mask = (torch.rand(b, 1, h, w, device=state.device) < self.fire_rate).float()
            else:
                fire_mask = torch.ones(b, 1, h, w, device=state.device)
            state = state + delta * fire_mask

        logits = state[:, : self.num_classes]
        return logits


def build_model():
    return CorroNCA()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()))
