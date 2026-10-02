"""VP-CorrosionNet: physics-guided, lightweight corrosion network operating on a single RGB image.

Pipeline (see docs/MODEL_ARCHITECTURE.md and Figure 1):
  sRGB -> linearize -> OpticalBranch (small encoder-decoder) -> LatentOpticsHead (Z)
       -> PhysicsStokesLayer (S_hat) -> VirtualAnalyzer stack -> PSRF -> PolFeatureBuilder
       -> fusion {early|mid|late} with colour/texture features -> CorrosionBackbone
       -> heads: segmentation, PittingHead, SeverityHead, aleatoric log-variance.
Every optical quantity produced here is an ESTIMATE (hat); see docs/LIMITATIONS.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..color import color_channels, color_features, linear_to_srgb, luminance, srgb_to_linear
from ..corrosion.pitting import N_CUES, PittingHead, pit_cue_maps
from ..corrosion.segmentation import NUM_CLASSES, sobel_mag
from ..corrosion.severity import SeverityHead
from ..geometry.curvature import normal_discontinuity
from ..inverse.latent_optics import LatentOpticsHead, PhysicsStokesLayer
from ..inverse.roughness import local_std, multiscale_laplacian_energy
from ..polarization.psrf import PSRF
from ..polarization.virtual_analyzer import CANONICAL_ANGLES_DEG, VirtualAnalyzer
from .blocks import DWSep, ECA, Decoder, Encoder, ConvBNAct

Tensor = torch.Tensor


@dataclass
class NetConfig:
    num_classes: int = NUM_CLASSES
    n_severity_levels: int = 5
    color_spaces: Sequence[str] = ("rgb", "lab")          # any subset of rgb, lab, hsv
    texture: bool = True
    # physics-derived feature groups (ablation switches)
    specdiff: bool = True
    vstokes: bool = True
    analyzer: bool = True
    geometry: bool = True
    roughness: bool = True
    psrf: bool = True
    uncertainty: bool = True
    coords: bool = False
    cartridge_prior: bool = False          # feed cylinder-prior normals to the optical branch
    psrf_order: int = 1
    stack_angles: Sequence[float] = CANONICAL_ANGLES_DEG
    analyzer_feature_angles: Sequence[float] = (0.0, 45.0, 90.0, 135.0)
    illumination: str = "environment"      # 'environment' | 'directional'
    base_material: str = "copper"
    residual_gain: float = 0.0             # learned residual polarization (0 = pure physics)
    fusion: str = "mid"                    # early | mid | late
    width_mult: float = 1.0
    dropout: float = 0.0
    edge_aware: bool = True
    feat_ch: int = 24
    optical_width: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "NetConfig":
        d = dict(d)
        for k in ("color_spaces", "stack_angles", "analyzer_feature_angles"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def needs_optics(self) -> bool:
        return any([self.specdiff, self.vstokes, self.analyzer, self.geometry, self.roughness, self.psrf,
                    self.uncertainty, self.cartridge_prior])

    def main_channels(self) -> int:
        return color_channels(self.color_spaces) + (2 if self.texture else 0) + (2 if self.coords else 0)

    def pol_channels(self) -> int:
        c = 0
        c += 2 if self.specdiff else 0
        c += 3 if self.vstokes else 0
        c += len(self.analyzer_feature_angles) if self.analyzer else 0
        c += 4 if self.geometry else 0
        c += 1 if self.roughness else 0
        c += (3 if self.psrf_order == 1 else 5) if self.psrf else 0
        c += 2 if self.uncertainty else 0
        return c


def _w(c: int, m: float) -> int:
    return max(8, int(round(c * m / 4)) * 4)


class CorrosionBackbone(nn.Module):
    """Encoder-decoder with early / mid / late fusion of the 'main' (appearance) and 'pol' (physics) inputs."""

    def __init__(self, in_main: int, in_pol: int, fusion: str, feat_ch: int, width_mult: float, edge_aware: bool,
                 num_classes: int, p_drop: float = 0.0):
        super().__init__()
        self.fusion = fusion if in_pol > 0 else "early"
        widths = tuple(_w(c, width_mult) for c in (16, 24, 40, 64))
        if self.fusion == "early":
            self.enc = Encoder(in_main + in_pol, widths, p_drop=p_drop)
            self.dec = Decoder(self.enc.out_channels, feat_ch, edge_aware)
            self.seg = nn.Sequential(DWSep(feat_ch, feat_ch), nn.Conv2d(feat_ch, num_classes, 1))
        elif self.fusion == "mid":
            pw = tuple(max(8, w // 2) for w in widths)
            self.enc_m, self.enc_p = Encoder(in_main, widths, p_drop=p_drop), Encoder(in_pol, pw, p_drop=p_drop)
            self.mix = nn.ModuleList([nn.Sequential(nn.Conv2d(a + b, a, 1, bias=False), nn.BatchNorm2d(a), nn.ReLU6(inplace=True), ECA(a))
                                      for a, b in zip(widths, pw)])
            self.dec = Decoder(widths, feat_ch, edge_aware)
            self.seg = nn.Sequential(DWSep(feat_ch, feat_ch), nn.Conv2d(feat_ch, num_classes, 1))
        elif self.fusion == "late":
            self.enc_m, self.enc_p = Encoder(in_main, widths, p_drop=p_drop), Encoder(in_pol, widths, p_drop=p_drop)
            self.dec_m, self.dec_p = Decoder(widths, feat_ch, edge_aware), Decoder(widths, feat_ch, edge_aware)
            self.seg_m, self.seg_p = nn.Conv2d(feat_ch, num_classes, 1), nn.Conv2d(feat_ch, num_classes, 1)
            self.fuse_logits = nn.Conv2d(2 * num_classes, num_classes, 1)
            self.fuse_feat = nn.Conv2d(2 * feat_ch, feat_ch, 1)
        else:
            raise ValueError(self.fusion)

    def forward(self, main: Tensor, pol: Optional[Tensor], edge: Tensor):
        if self.fusion == "early":
            x = main if pol is None else torch.cat([main, pol], 1)
            f = self.dec(self.enc(x), edge)
            return f, self.seg(f)
        if self.fusion == "mid":
            fm, fp = self.enc_m(main), self.enc_p(pol)
            fused = [m(torch.cat([a, b], 1)) for m, a, b in zip(self.mix, fm, fp)]
            f = self.dec(fused, edge)
            return f, self.seg(f)
        fm, fp = self.dec_m(self.enc_m(main), edge), self.dec_p(self.enc_p(pol), edge)
        lm, lp = self.seg_m(fm), self.seg_p(fp)
        return self.fuse_feat(torch.cat([fm, fp], 1)), self.fuse_logits(torch.cat([lm, lp], 1))


class OpticalBranch(nn.Module):
    """RGB (+ optional normal prior) -> latent optical field Z -> S_hat."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        in_ch = 3 + (3 if cfg.cartridge_prior else 0)
        widths = tuple(_w(c, cfg.optical_width) for c in (16, 24, 40, 64))
        self.enc = Encoder(in_ch, widths, depths=(1, 1, 2))
        self.dec = Decoder(widths, 16, edge_aware=False)
        self.head = LatentOpticsHead(16, normal_prior_weight=1.0 if cfg.cartridge_prior else 0.0)
        self.phys = PhysicsStokesLayer(cfg.base_material, cfg.illumination, cfg.residual_gain)
        self.cfg = cfg

    def forward(self, srgb: Tensor, normal_prior: Optional[Tensor]) -> Dict[str, Tensor]:
        x = srgb if not self.cfg.cartridge_prior else torch.cat([srgb, normal_prior if normal_prior is not None else torch.zeros_like(srgb)], 1)
        f = self.dec(self.enc(x))
        f = F.interpolate(f, size=srgb.shape[-2:], mode="bilinear", align_corners=False)
        z = self.head(f, normal_prior if self.cfg.cartridge_prior else None)
        z["S_hat"] = self.phys(z)
        return z


class VPCorrosionNet(nn.Module):
    def __init__(self, cfg: NetConfig | dict | None = None):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, NetConfig) else NetConfig.from_dict(cfg or {})
        c = self.cfg
        self.optics = OpticalBranch(c) if c.needs_optics else None
        self.analyzer = VirtualAnalyzer()
        self.psrf = PSRF(c.stack_angles, c.psrf_order) if (c.psrf and c.needs_optics) else None
        self.backbone = CorrosionBackbone(c.main_channels(), c.pol_channels() if c.needs_optics else 0, c.fusion, c.feat_ch,
                                          c.width_mult, c.edge_aware, c.num_classes, c.dropout)
        self.pit_head = PittingHead(c.feat_ch)
        self.sev_head = SeverityHead(c.feat_ch, c.n_severity_levels)
        self.unc_head = nn.Conv2d(c.feat_ch, 1, 1)
        self.distill_proj = nn.Conv2d(c.feat_ch, c.feat_ch, 1)
        nn.init.constant_(self.unc_head.bias, -2.0)

    # --- feature construction -------------------------------------------------------------------
    def _pol_features(self, z: Dict[str, Tensor], lum_lin: Tensor) -> tuple[Tensor, dict]:
        c, info = self.cfg, {}
        S = z["S_hat"]                                                    # (B,3,3,H,W)
        w = torch.tensor([0.2126729, 0.7151522, 0.0721750], device=S.device, dtype=S.dtype).view(1, 3, 1, 1, 1)
        SL = (S * w).sum(1, keepdim=True)                                 # (B,1,3,H,W) luminance Stokes
        stack = self.analyzer(SL, c.stack_angles)[:, :, 0]                # (B,A,H,W)
        info["stack"] = stack
        feats = []
        if c.specdiff:
            Sl, Dl = luminance(z["S"]), luminance(z["D"])
            sp = Sl / (Sl + Dl).clamp_min(1e-4)
            feats += [sp, 1 - sp]
        coef = self.psrf.fit(stack) if self.psrf is not None else None
        if coef is not None:
            info["psrf_coef"] = coef
        mag = torch.sqrt(SL[:, 0, 1] ** 2 + SL[:, 0, 2] ** 2 + 1e-8)
        if c.vstokes:
            dolp = (mag / SL[:, 0, 0].clamp_min(1e-4)).clamp(0, 1)
            feats += [dolp.unsqueeze(1), (SL[:, 0, 1] / mag).unsqueeze(1), (SL[:, 0, 2] / mag).unsqueeze(1)]
        if c.analyzer:
            idx = [min(range(len(c.stack_angles)), key=lambda i: abs(c.stack_angles[i] - a)) for a in c.analyzer_feature_angles]
            feats.append(stack[:, idx])
        if c.geometry:
            n = z["n"]
            feats += [n, normal_discontinuity(n) / math.pi]
        if c.roughness:
            feats.append(z["r"])
        if c.psrf and coef is not None:
            feats.append(coef[:, :3].clamp(-4, 4) if c.psrf_order == 1 else coef[:, :5].clamp(-4, 4))
        if c.uncertainty:
            feats += [z["u"], z["g"]]
        info["S_lum"] = SL
        return torch.cat(feats, 1), info

    def _main_features(self, srgb: Tensor) -> Tensor:
        c = self.cfg
        f = [color_features(srgb, c.color_spaces)]
        if c.texture:
            lum = srgb.mean(1, keepdim=True)
            f += [local_std(lum, 7) * 4, multiscale_laplacian_energy(lum) * 4]
        if c.coords:
            B, _, H, W = srgb.shape
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=srgb.device), torch.linspace(-1, 1, W, device=srgb.device), indexing="ij")
            f.append(torch.stack([xs, ys]).unsqueeze(0).expand(B, -1, -1, -1))
        return torch.cat(f, 1)

    # --- forward ------------------------------------------------------------------------------------
    def forward(self, rgb: Tensor, normals_prior: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """rgb: sRGB-encoded (B,3,H,W) in [0,1]; H, W divisible by 16. normals_prior: (B,3,H,W) cylinder-prior normals."""
        lin = srgb_to_linear(rgb)
        lum_lin = luminance(lin)
        out: Dict[str, Tensor] = {"lin": lin}
        pol = None
        if self.optics is not None:
            z = self.optics(rgb, normals_prior)
            pol, info = self._pol_features(z, lum_lin)
            out.update({"latent": z, "S_hat": z["S_hat"], "lin_rec": z["D"] + z["S"], **info})
        main = self._main_features(rgb)
        edge = sobel_mag(rgb.mean(1, keepdim=True))
        feat, logits = self.backbone(main, pol, edge)
        size = rgb.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        normals = out["latent"]["n"] if self.optics is not None else None
        cues = pit_cue_maps(lum_lin, normals)
        pit = self.pit_head(feat, cues)
        seg_prob_half = torch.softmax(logits, 1)
        sev = self.sev_head(feat, seg_prob_half, torch.sigmoid(pit))
        out.update({
            "seg_logits": up(logits), "pit_logit": up(pit), "seg_logvar": up(self.unc_head(feat)),
            "feat": feat, "feat_proj": self.distill_proj(feat), **sev,
        })
        return out

    # --- counterfactual polarimetric ensemble ------------------------------------------------------------
    @torch.no_grad()
    def cpe(self, out: dict, K: int = 16, angles_deg: Sequence[float] = CANONICAL_ANGLES_DEG, level: float = 0.95) -> dict:
        from ..polarization.uncertainty import counterfactual_polarimetric_ensemble
        if self.optics is None:
            raise RuntimeError("model has no optical branch")
        z = out["latent"]
        sampler = lambda: self.optics.phys(self.optics.head.sample(z))
        return counterfactual_polarimetric_ensemble(sampler, self.analyzer, angles_deg, K, level)
