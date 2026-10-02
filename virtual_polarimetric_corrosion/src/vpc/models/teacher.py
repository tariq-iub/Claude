"""Hardware teacher: consumes REAL (or, in smoke tests, synthetic stand-in) analyzer images.

Input stack (B, A, 3, H, W) of measured analyzer images in linear radiometric units (A >= 3, angles in degrees).
Measured Stokes are computed analytically (least squares); the learned part predicts corrosion from
colour (S0 image) + measured polarimetric features. Outputs include the measured polarimetric representation used
as distillation targets for the RGB-only student.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..color import color_channels, color_features, linear_to_srgb, luminance
from ..corrosion.pitting import PittingHead, pit_cue_maps
from ..corrosion.segmentation import NUM_CLASSES, sobel_mag
from ..corrosion.severity import SeverityHead
from ..geometry.curvature import normal_discontinuity
from ..inverse.roughness import local_std, multiscale_laplacian_energy
from ..optics.stokes import fit_linear_stokes
from .blocks import DWSep
from .vp_corrosion_net import CorrosionBackbone, NetConfig

Tensor = torch.Tensor


def measured_stokes(stack: Tensor, angles_deg: Sequence[float]) -> Tensor:
    """(B,A,3,H,W) -> S (B,3[colour],3[S0,S1,S2],H,W) by least squares over analyzer angles."""
    th = torch.as_tensor(angles_deg, dtype=stack.dtype, device=stack.device) * (math.pi / 180.0)
    S = fit_linear_stokes(stack, th, dim=1)          # (B,3[S],3[colour],H,W)
    return S.permute(0, 2, 1, 3, 4).contiguous()


class PolarimetricTeacher(nn.Module):
    def __init__(self, angles_deg: Sequence[float] = (0, 45, 90, 135), cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        self.angles = tuple(angles_deg)
        c = self.cfg
        self.n_stack = min(len(self.angles), 4)
        in_pol = 3 + 3 + self.n_stack + 1  # dolp,cos,sin | a0,a1,b1 | stack lum | normal-discontinuity proxy (zeros: no normals)
        self.backbone = CorrosionBackbone(color_channels(c.color_spaces) + 2, in_pol, c.fusion, c.feat_ch, c.width_mult, c.edge_aware, c.num_classes)
        self.pit_head = PittingHead(c.feat_ch)
        self.sev_head = SeverityHead(c.feat_ch, c.n_severity_levels)

    def forward(self, stack: Tensor) -> Dict[str, Tensor]:
        S = measured_stokes(stack, self.angles)                       # (B,3,3,H,W)
        w = torch.tensor([0.2126729, 0.7151522, 0.0721750], device=S.device, dtype=S.dtype).view(1, 3, 1, 1, 1)
        SL = (S * w).sum(1)                                           # (B,3,H,W)
        s0_rgb = S[:, :, 0].clamp(0, 1)                               # unpolarized-equivalent image
        srgb = linear_to_srgb(s0_rgb)
        mag = torch.sqrt(SL[:, 1] ** 2 + SL[:, 2] ** 2 + 1e-8)
        dolp = (mag / SL[:, 0].clamp_min(1e-4)).clamp(0, 1).unsqueeze(1)
        feats = [dolp, (SL[:, 1] / mag).unsqueeze(1), (SL[:, 2] / mag).unsqueeze(1), 0.5 * SL[:, 0:1].clamp(-4, 4),
                 0.5 * SL[:, 1:2].clamp(-4, 4), 0.5 * SL[:, 2:3].clamp(-4, 4)]
        wl = torch.tensor([0.2126729, 0.7151522, 0.0721750], device=S.device, dtype=S.dtype).view(1, 1, 3, 1, 1)
        lum_stack = (stack * wl).sum(2)                                # (B,A,H,W)
        feats.append(lum_stack[:, : self.n_stack])
        feats.append(torch.zeros_like(dolp))
        pol = torch.cat(feats, 1)
        lum = srgb.mean(1, keepdim=True)
        main = torch.cat([color_features(srgb, self.cfg.color_spaces), local_std(lum, 7) * 4, multiscale_laplacian_energy(lum) * 4], 1)
        feat, logits = self.backbone(main, pol, sobel_mag(lum))
        cues = pit_cue_maps(luminance(s0_rgb), None)
        pit = self.pit_head(feat, cues)
        sev = self.sev_head(feat, torch.softmax(logits, 1), torch.sigmoid(pit))
        size = stack.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return {"S_meas": S, "dolp_meas": dolp, "seg_logits": up(logits), "pit_logit": up(pit), "feat": feat, **sev}
