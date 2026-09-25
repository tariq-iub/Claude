"""
Corro-SSM: Perceptually-Gated State-Space Corrosion Model
=============================================================
Core primitive: a linear diagonal state-space recurrence scanned separably
along rows then columns (2D-separable SSM, O(HW) complexity, no attention
matrix), where the *state-retention gate* at every step is derived from the
CIEDE2000 colour distance to the previous pixel in the scan:

    g_t = sigmoid( -ΔE00(lab_t, lab_{t-1}) / kappa )
    h_t = g_t * (A ⊙ h_{t-1}) + (1 - g_t) * (B x_t)
    y_t = C h_t + D x_t

This ties the *forget gate* of the recurrence directly to a physically
motivated quantity (a strong perceptual colour jump = a material boundary =
state should reset), which is fundamentally different from Mamba/S4, whose
gates are purely learned functions of the input embedding with no explicit
colour-metric coupling.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, rgb_to_hsv, deltaE2000
from common.dataset import NUM_CLASSES


class PerceptualGatedSSM1D(nn.Module):
    """Applies the gated diagonal SSM recurrence along the last spatial axis (assumed sequence dim)."""

    def __init__(self, dim: int, kappa_init: float = 10.0):
        super().__init__()
        self.dim = dim
        self.log_a = nn.Parameter(torch.zeros(dim))  # A = sigmoid(log_a) in (0,1): stable diagonal decay
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.C = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.D = nn.Parameter(torch.ones(dim) * 0.5)
        self.log_kappa = nn.Parameter(torch.tensor(kappa_init).log())

    def forward(self, x_seq: torch.Tensor, lab_seq: torch.Tensor):
        # x_seq: (B, L, D); lab_seq: (B, L, 3)
        b, L, d = x_seq.shape
        a = torch.sigmoid(self.log_a)  # (D,)
        bx = torch.einsum("bld,de->ble", x_seq, self.B)

        de = deltaE2000(lab_seq[:, 1:], lab_seq[:, :-1])  # (B, L-1)
        kappa = self.log_kappa.exp().clamp(min=1e-3)
        gate = torch.sigmoid(-de / kappa)  # (B, L-1) in (0,1); ~0 at strong colour boundary
        gate = F.pad(gate, (1, 0), value=1.0).unsqueeze(-1)  # (B,L,1), first step no history

        h = torch.zeros(b, d, device=x_seq.device, dtype=x_seq.dtype)
        outs = []
        for t in range(L):
            h = gate[:, t] * (a * h) + (1 - gate[:, t]) * bx[:, t]
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)  # (B,L,D)
        y = torch.einsum("bld,de->ble", h_seq, self.C) + self.D * x_seq
        return y


class SeparableSSMBlock(nn.Module):
    """Scans rows (left->right) then columns (top->bottom), each with its own gated SSM."""

    def __init__(self, dim: int):
        super().__init__()
        self.row_ssm = PerceptualGatedSSM1D(dim)
        self.col_ssm = PerceptualGatedSSM1D(dim)
        self.norm = nn.GroupNorm(4, dim)

    def forward(self, feat: torch.Tensor, lab: torch.Tensor):
        b, c, h, w = feat.shape
        # row scan
        x_rows = feat.permute(0, 2, 3, 1).reshape(b * h, w, c)
        lab_rows = lab.permute(0, 2, 3, 1).reshape(b * h, w, 3)
        y_rows = self.row_ssm(x_rows, lab_rows).reshape(b, h, w, c).permute(0, 3, 1, 2)

        # column scan
        x_cols = y_rows.permute(0, 3, 2, 1).reshape(b * w, h, c)
        lab_cols = lab.permute(0, 3, 2, 1).reshape(b * w, h, 3)
        y_cols = self.col_ssm(x_cols, lab_cols).reshape(b, w, h, c).permute(0, 3, 2, 1)

        return feat + self.norm(y_cols)


class CorroSSM(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dim: int = 20, n_blocks: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(9, dim, 1)
        self.blocks = nn.ModuleList([SeparableSSMBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Conv2d(dim, num_classes, 1)

    def forward(self, rgb: torch.Tensor):
        lab = rgb_to_lab(rgb)
        hsv = rgb_to_hsv(rgb)
        x = torch.cat([rgb, lab, hsv], dim=1)
        feat = self.stem(x)
        for blk in self.blocks:
            feat = blk(feat, lab)
        return self.head(feat)


def build_model():
    return CorroSSM()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(1, 3, 32, 32)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()))
