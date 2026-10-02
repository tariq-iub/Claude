"""Light-weight building blocks: depthwise-separable conv, inverted-residual bottleneck, ECA attention, edge gate."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class ConvBNAct(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, groups=1, act=True):
        layers = [nn.Conv2d(i, o, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(o)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class DWSep(nn.Sequential):
    def __init__(self, i, o, s=1):
        super().__init__(ConvBNAct(i, i, 3, s, groups=i), ConvBNAct(i, o, 1))


class InvertedResidual(nn.Module):
    """MobileNetV2-style residual bottleneck."""

    def __init__(self, i, o, s=1, expand=4, p_drop=0.0):
        super().__init__()
        h = i * expand
        self.use_res = s == 1 and i == o
        self.block = nn.Sequential(ConvBNAct(i, h, 1), ConvBNAct(h, h, 3, s, groups=h), ConvBNAct(h, o, 1, act=False))
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.drop(self.block(x))
        return x + y if self.use_res else y


class ECA(nn.Module):
    """Efficient channel attention (Wang et al. 2020): 1-D conv over pooled channel descriptor; ~k parameters."""

    def __init__(self, ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        w = x.mean((2, 3)).unsqueeze(1)
        w = torch.sigmoid(self.conv(w)).transpose(1, 2).unsqueeze(-1)
        return x * w


class EdgeGate(nn.Module):
    """Edge-aware skip gating: skip * (1 + sigmoid(conv(edge))) with edge = Sobel magnitude of luminance."""

    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Conv2d(1, ch, 1)

    def forward(self, skip, edge):
        e = F.interpolate(edge, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * (1 + torch.sigmoid(self.proj(e)))


class Encoder(nn.Module):
    """Stem (stride 2) + 3 stages (stride 2 each) -> features at 1/2, 1/4, 1/8, 1/16."""

    def __init__(self, in_ch, widths=(16, 24, 40, 64), depths=(1, 2, 2), expand=3, p_drop=0.0):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, widths[0], 3, 2), DWSep(widths[0], widths[0]))
        stages, prev = [], widths[0]
        for w, d in zip(widths[1:], depths):
            blocks = [InvertedResidual(prev, w, 2, expand, p_drop)] + [InvertedResidual(w, w, 1, expand, p_drop) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks))
            prev = w
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(widths)

    def forward(self, x):
        feats = [self.stem(x)]
        for s in self.stages:
            feats.append(s(feats[-1]))
        return feats


class Decoder(nn.Module):
    """Multi-scale edge-aware decoder; returns features at 1/2 resolution."""

    def __init__(self, enc_ch, out_ch=24, edge_aware=True):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in enc_ch])
        self.gates = nn.ModuleList([EdgeGate(out_ch) if edge_aware else nn.Identity() for _ in enc_ch[:-1]])
        self.fuse = nn.ModuleList([DWSep(out_ch, out_ch) for _ in enc_ch[:-1]])
        self.att = ECA(out_ch)
        self.edge_aware = edge_aware

    def forward(self, feats, edge=None):
        x = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            skip = self.lat[i](feats[i])
            if self.edge_aware and edge is not None:
                skip = self.gates[i](skip, edge)
            x = self.fuse[i](x + skip)
        return self.att(x)
