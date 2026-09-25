"""
HyperCorNet: Hypergraph Corrosion Recognition Network
=======================================================
Core primitive: pixels are grouped into soft superpixel "cells" by a
differentiable relaxation of SLIC (joint spatial + CIELAB assignment), and
a *hypergraph* is built whose hyperedges are (a) each cell's own membership
(local hyperedges) and (b) global colour-similarity bins that connect
non-adjacent cells sharing similar chroma/hue (long-range hyperedges),
letting information flow between disjoint corrosion patches of the same
oxidation type without an explicit graph-attention softmax over all pairs.
Hypergraph convolution (Feng et al. formalism) then updates cell features,
which are scattered back to pixels for the segmentation head.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.colorspace import rgb_to_lab, spatial_coords
from common.dataset import NUM_CLASSES


class SoftSLIC(nn.Module):
    """
    Differentiable soft-SLIC: places a grid of M = gh*gw cell centers, and
    computes soft assignment of every pixel to every cell using a joint
    spatial + Lab distance:

        a_{i,m} = softmax_m( -( d_s(i,m)^2 / sigma_s^2 + d_c(i,m)^2 / sigma_c^2 ) )

    Cell centers' Lab coordinate is re-estimated as the assignment-weighted
    mean each forward pass (one differentiable Lloyd iteration), giving a
    lightweight amortised clustering that trains jointly with the rest of
    the network instead of running a separate SLIC pre-processing pass.
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 8):
        super().__init__()
        self.gh, self.gw = grid_h, grid_w
        self.log_sigma_s = nn.Parameter(torch.tensor(0.3).log())
        self.log_sigma_c = nn.Parameter(torch.tensor(8.0).log())

    def forward(self, lab: torch.Tensor, xy: torch.Tensor):
        b, _, h, w = lab.shape
        m = self.gh * self.gw
        cy = torch.linspace(-1, 1, self.gh, device=lab.device)
        cx = torch.linspace(-1, 1, self.gw, device=lab.device)
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        centers_xy = torch.stack([gx, gy], dim=-1).reshape(m, 2)  # (M,2)

        pix_xy = xy.permute(0, 2, 3, 1).reshape(b, h * w, 2)  # (B,N,2)
        pix_lab = lab.permute(0, 2, 3, 1).reshape(b, h * w, 3)  # (B,N,3)

        d_s2 = torch.cdist(pix_xy, centers_xy.unsqueeze(0).expand(b, -1, -1)) ** 2  # (B,N,M)

        # initial cell Lab = nearest-center pooled mean (one soft Lloyd step)
        sigma_s2 = (self.log_sigma_s.exp() ** 2).clamp(min=1e-3)
        init_logits = -d_s2 / sigma_s2
        init_assign = F.softmax(init_logits, dim=2)  # (B,N,M)
        cell_lab = torch.einsum("bnm,bnc->bmc", init_assign, pix_lab) / (
            init_assign.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6
        )

        d_c2 = torch.cdist(pix_lab, cell_lab) ** 2  # (B,N,M)
        sigma_c2 = (self.log_sigma_c.exp() ** 2).clamp(min=1e-3)
        logits = -d_s2 / sigma_s2 - d_c2 / sigma_c2
        assign = F.softmax(logits, dim=2)  # (B,N,M) soft incidence H (pixel-to-cell)
        return assign, cell_lab  # assign: (B,N,M)


class HypergraphConv(nn.Module):
    """
    Two-stage hypergraph convolution:
      1. local incidence (pixel<->cell, from SoftSLIC) aggregates pixel features into cell features
      2. global colour-similarity hyperedges connect cells with similar mean Lab,
         built as soft membership to `n_bins` learned colour prototypes (a second incidence matrix)
      3. cell features are updated by hypergraph propagation:
             X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Theta
      4. updated cell features are scattered back to pixels via the same incidence.
    """

    def __init__(self, in_dim: int, out_dim: int, n_bins: int = 6):
        super().__init__()
        self.theta_local = nn.Linear(in_dim, out_dim)
        self.theta_global = nn.Linear(out_dim, out_dim)
        self.color_bins = nn.Parameter(torch.randn(n_bins, 3) * 10.0)  # learned Lab prototypes for global hyperedges

    def forward(self, pix_feat: torch.Tensor, assign: torch.Tensor, cell_lab: torch.Tensor):
        # pix_feat: (B,N,Din), assign (H, pixel-cell incidence): (B,N,M)
        b, n, din = pix_feat.shape
        m = assign.shape[-1]

        d_v = assign.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (B,N,1) pixel degree
        d_e = assign.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,1,M) cell(hyperedge) degree

        # local aggregation: cell_feat = H^T D_v^{-1/2} X
        norm_pix = pix_feat / d_v.sqrt()
        cell_feat = torch.einsum("bnm,bnd->bmd", assign, norm_pix) / d_e.transpose(1, 2).sqrt()
        cell_feat = self.theta_local(cell_feat)

        # global colour hyperedges: soft membership of cells to colour bins
        bin_logits = -torch.cdist(cell_lab, self.color_bins.unsqueeze(0).expand(b, -1, -1)) ** 2 / 400.0
        bin_assign = F.softmax(bin_logits, dim=2)  # (B,M,K) second incidence matrix H2
        bin_deg = bin_assign.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,1,K)
        bin_feat = torch.einsum("bmk,bmd->bkd", bin_assign, cell_feat) / bin_deg.transpose(1, 2)
        cell_feat_global = torch.einsum("bmk,bkd->bmd", bin_assign, bin_feat)
        cell_feat = cell_feat + self.theta_global(cell_feat_global)
        cell_feat = F.gelu(cell_feat)

        # scatter back to pixels: X_pix' = H D_v^{-1/2} D_e^{-1} cell_feat  (renormalised)
        pix_out = torch.einsum("bnm,bmd->bnd", assign, cell_feat) / d_v
        return pix_out


class HyperCorNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, feat_dim: int = 24, grid: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(8, feat_dim, 3, padding=1), nn.GroupNorm(4, feat_dim), nn.GELU(),
        )
        self.slic = SoftSLIC(grid, grid)
        self.hconv1 = HypergraphConv(feat_dim, feat_dim)
        self.hconv2 = HypergraphConv(feat_dim, feat_dim)
        self.head = nn.Conv2d(feat_dim, num_classes, 1)

    def forward(self, rgb: torch.Tensor):
        b, _, h, w = rgb.shape
        lab = rgb_to_lab(rgb)
        xy = spatial_coords(h, w, rgb.device, rgb.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        stack = torch.cat([rgb, lab, xy], dim=1)  # 3+3+2=8
        feat = self.stem(stack)

        assign, cell_lab = self.slic(lab, xy)
        pix_feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        pix_feat = pix_feat + self.hconv1(pix_feat, assign, cell_lab)
        pix_feat = pix_feat + self.hconv2(pix_feat, assign, cell_lab)
        feat_out = pix_feat.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.head(feat_out)


def build_model():
    return HyperCorNet()


if __name__ == "__main__":
    m = build_model()
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    print(y.shape, sum(p.numel() for p in m.parameters()))
