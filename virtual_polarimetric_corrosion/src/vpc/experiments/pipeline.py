"""Config -> datasets/models helpers and the synthetic physics-validation harness used by the scripts."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from ..models.vp_corrosion_net import NetConfig
from ..optics.stokes import stokes_from_four, fit_linear_stokes
from ..polarization.psrf import PSRF
from ..polarization.virtual_analyzer import CANONICAL_ANGLES_DEG, VirtualAnalyzer
from . import metrics as M
from .datasets import RealManifestDataset, SyntheticCorrosionDataset
from .splits import group_stratified_split
from .synthetic import SynthConfig, make_group, render_view
from .train import TrainConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("extends")))
        for k, v in cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        cfg = base
    return cfg


def net_config(cfg: dict) -> NetConfig:
    return NetConfig.from_dict(cfg.get("net", {}))


def train_config(cfg: dict, seed: int | None = None) -> TrainConfig:
    t = dict(cfg.get("train", {}))
    t["seed"] = cfg.get("seed", 0) if seed is None else seed
    return TrainConfig(**t)


def build_splits(cfg: dict):
    """Returns (train_ds, val_ds, test_ds, info). Splits are by cartridge GROUP, never by pixel or view."""
    d = cfg["data"]
    if d["type"] == "synthetic":
        sc = SynthConfig(size=d["size"])
        full = SyntheticCorrosionDataset(range(d["n_groups"]), d["views_per_group"], sc, seed=cfg.get("seed", 0))
        sp = group_stratified_split(full.group_ids(), full.severity_levels(), tuple(d["split_ratios"]), d["split_seed"])
        groups = full.group_ids()
        mk = lambda idx: SyntheticCorrosionDataset(sorted(set(groups[idx].tolist())), d["views_per_group"], sc, seed=cfg.get("seed", 0))
        return mk(sp["train"]), mk(sp["val"]), mk(sp["test"]), {"origin": "SYNTHETIC", "n_groups": d["n_groups"]}
    import pandas as pd
    df = pd.read_csv(d["manifest"])
    strata = df["severity_level"].values if "severity_level" in df else np.zeros(len(df), int)
    sp = group_stratified_split(df["group_id"].values, strata, tuple(d["split_ratios"]), d["split_seed"])
    mk = lambda idx: RealManifestDataset(d["manifest"], d.get("root", "."), indices=idx)
    return mk(sp["train"]), mk(sp["val"]), mk(sp["test"]), {"origin": "REAL", "n_groups": int(df["group_id"].nunique())}


# ------------------------------------------------------------------ synthetic validation of the virtual-physics modules
def synthetic_validation(n_samples: int = 24, size: int = 96, seed: int = 0) -> List[dict]:
    """Quantitative unit validation against renderer ground truth. SYNTHETIC ONLY. Rows: (check, value, tolerance_or_reference, note).
    Includes the physics-only virtual polarizer (no learning) as the reference for how far untrained physics gets from RGB."""
    from ..models.baseline import physics_only_virtual_polarizer
    from ..optics.fresnel import fresnel_reflectance, preset
    from ..optics.reflection import diffuse_dolp, roughness_depolarization
    rng = np.random.default_rng(seed)
    cfg = SynthConfig(size=size, read_noise=0.0, shot_noise=0.0, exposure_range=(0.2, 0.3), sensor_clip=False)   # unclipped, noise-free
    va = VirtualAnalyzer()
    errs = {"stack_vs_malus": [], "four_angle_stokes": [], "ls_stokes_8": [], "psrf_order1_resid": [], "psrf_order2_ho_energy": [],
            "periodicity": [], "orthogonal_sum": []}
    keys = [(v, m) for v in ("physics_only_untrained", "oracle_split_prior_normals") for m in ("environment", "directional")]
    pol = {k: {"dolp_mae": [], "aolp_err_deg": [], "cos2": []} for k in keys}
    n0, k0 = preset("copper")
    L = lambda x: 0.2126729 * x[0] + 0.7151522 * x[1] + 0.0721750 * x[2]
    for i in range(n_samples):
        grp = make_group(int(rng.integers(1e6)), cfg)
        d = render_view(grp, int(rng.integers(1e6)), cfg)
        S, stack, ang = d["S_gt"], d["stack"], d["angles"]
        st = va(S.unsqueeze(0), ang.tolist())[0]
        errs["stack_vs_malus"].append(float((st - stack).abs().max()))
        idx = [int(np.argmin(np.abs(ang.numpy() - a))) for a in (0, 45, 90, 135)]
        S4 = stokes_from_four(*[stack[j] for j in idx])
        errs["four_angle_stokes"].append(float((S4.permute(1, 0, 2, 3) - S).abs().max()))
        th = ang * math.pi / 180
        errs["ls_stokes_8"].append(float((fit_linear_stokes(stack, th, 0).permute(1, 0, 2, 3) - S).abs().max()))
        lum = stack.mean(1).unsqueeze(0)
        c1 = PSRF(ang.tolist(), 1).fit(lum)
        errs["psrf_order1_resid"].append(float((PSRF(ang.tolist(), 1).reconstruct(c1, ang.tolist()) - lum).abs().max()))
        c2 = PSRF(ang.tolist(), 2).fit(lum)
        errs["psrf_order2_ho_energy"].append(float(c2[:, 3:].abs().max()))
        errs["periodicity"].append(float((va(S.unsqueeze(0), [33.0]) - va(S.unsqueeze(0), [213.0])).abs().max()))
        errs["orthogonal_sum"].append(float((va(S.unsqueeze(0), [20.0]) + va(S.unsqueeze(0), [110.0]) - S[:, 0].unsqueeze(0).unsqueeze(0)).abs().max()))
        # both illumination modes of the SAME cartridge/pose, so the effect of the illumination assumption is isolated
        for mode in ("environment", "directional"):
            dm = render_view(grp, int(d["meta"]["view_seed"]), cfg, force_mode=mode)
            Sm, obj = dm["S_gt"], dm["obj_mask"].numpy()
            Lt = L(Sm.numpy())
            # (a) untrained physics-only baseline from RGB (environment-illumination assumption inside)
            r = physics_only_virtual_polarizer(dm["rgb"].unsqueeze(0), dm["normals_prior"].unsqueeze(0), 45.0)
            Lh = L(r["S_hat"][0].numpy()); Lh = Lh * (Lt[0][obj].mean() / max(Lh[0][obj].mean(), 1e-9))
            # (b) oracle: TRUE diffuse/specular split, cylinder-prior normals, environment-mode formulas (isolates the attribution error)
            lg, n = dm["latent_gt"], dm["normals_prior"]
            zen, az = torch.acos(n[2].clamp(-1, 1)), torch.atan2(n[1], n[0])
            Rs, Rp, _ = fresnel_reflectance(zen.unsqueeze(0), n0.view(3, 1, 1), k0.view(3, 1, 1))
            rho_s = (Rs - Rp) / (Rs + Rp) * roughness_depolarization(lg["rough"].unsqueeze(0))
            rho_d = diffuse_dolp(zen, lg["n_diff"])
            s1 = lg["S"] * rho_s * torch.cos(2 * (az + math.pi / 2)) + lg["D"] * rho_d * torch.cos(2 * az)
            s2 = lg["S"] * rho_s * torch.sin(2 * (az + math.pi / 2)) + lg["D"] * rho_d * torch.sin(2 * az)
            Lo = L(torch.stack([lg["S"] + lg["D"], s1, s2]).permute(1, 0, 2, 3).numpy())     # (colour, Stokes, H, W) -> luminance Stokes
            for name, Lx in (("physics_only_untrained", Lh), ("oracle_split_prior_normals", Lo)):
                a_h, a_t = M.aolp_np(Lx), M.aolp_np(Lt)
                w = M.dolp_np(Lt) * obj
                pol[(name, mode)]["dolp_mae"].append(float(np.abs(M.dolp_np(Lx) - M.dolp_np(Lt))[obj].mean()))
                pol[(name, mode)]["aolp_err_deg"].append(float(np.degrees(M.aolp_error(a_h, a_t, w))) if w.sum() > 0 else float("nan"))
                pol[(name, mode)]["cos2"].append(float((np.cos(2 * (a_h - a_t)) * w).sum() / max(w.sum(), 1e-12)) if w.sum() > 0 else float("nan"))
    rows = [{"check": k, "max_over_samples": float(np.max(v)), "mean": float(np.mean(v)), "reference": "exact up to float32 (<1e-4)",
             "n": n_samples, "data_origin": "SYNTHETIC"} for k, v in errs.items()]
    for (name, mode), dct in pol.items():
        for k, v in dct.items():
            rows.append({"check": f"{name}::{mode}::{k}", "max_over_samples": float(np.nanmax(v)), "mean": float(np.nanmean(v)),
                         "reference": "AoLP chance level (uniform wrapped error) = 45 deg; oracle isolates specular/diffuse attribution; mode isolates illumination assumption",
                         "n": n_samples, "data_origin": "SYNTHETIC"})
    return rows
