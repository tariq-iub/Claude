"""Ablation arms (cumulative, plus leave-one-out helpers) and runner. Results are written as RAW per-seed rows; never as
pre-filled numbers."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence

import numpy as np

from ..models.vp_corrosion_net import NetConfig, VPCorrosionNet
from .datasets import SyntheticCorrosionDataset
from .train import TrainConfig, evaluate, train

_OFF = dict(color_spaces=("rgb",), texture=False, specdiff=False, vstokes=False, analyzer=False, geometry=False, roughness=False,
            psrf=False, uncertainty=False, coords=False, cartridge_prior=False)

# (name, overrides applied cumulatively on top of the previous arm)
CUMULATIVE_STEPS = [
    ("rgb_only", {}),
    ("+lab", dict(color_spaces=("rgb", "lab"))),
    ("+hsv", dict(color_spaces=("rgb", "lab", "hsv"))),
    ("+texture", dict(texture=True)),
    ("+specular_diffuse", dict(specdiff=True)),
    ("+virtual_stokes", dict(vstokes=True)),
    ("+virtual_analyzer", dict(analyzer=True)),
    ("+geometry", dict(geometry=True)),
    ("+roughness", dict(roughness=True)),
    ("+psrf", dict(psrf=True)),
    ("+uncertainty_feats", dict(uncertainty=True)),
    ("+cartridge_prior", dict(cartridge_prior=True)),
]
# 'distillation' is a TRAINING option (needs hardware teacher), handled via run_ablation(distill=True)
COLOR_SPACE_ARMS = {"cs_rgb": ("rgb",), "cs_lab": ("lab",), "cs_hsv": ("hsv",), "cs_rgb+lab": ("rgb", "lab"), "cs_rgb+hsv": ("rgb", "hsv"),
                    "cs_rgb+lab+hsv": ("rgb", "lab", "hsv")}
FUSION_ARMS = ("early", "mid", "late")


def cumulative_configs(base: dict | None = None) -> Dict[str, NetConfig]:
    d = {**_OFF, **(base or {})}
    out = {}
    for name, ov in CUMULATIVE_STEPS:
        d = {**d, **ov}
        out[name] = NetConfig.from_dict(d)
    return out


def leave_one_out_configs(full: NetConfig) -> Dict[str, NetConfig]:
    """Full model minus one physics feature group at a time."""
    groups = ["specdiff", "vstokes", "analyzer", "geometry", "roughness", "psrf", "uncertainty", "texture"]
    out = {"full": full}
    for g in groups:
        d = full.to_dict(); d[g] = False
        out[f"full-{g}"] = NetConfig.from_dict(d)
    d = full.to_dict(); d["color_spaces"] = ("rgb",); out["full-lab"] = NetConfig.from_dict(d)
    d = full.to_dict(); d["cartridge_prior"] = False; out["full-cartridge_prior"] = NetConfig.from_dict(d)
    return out


def color_space_configs(base: NetConfig) -> Dict[str, NetConfig]:
    out = {}
    for k, cs in COLOR_SPACE_ARMS.items():
        d = base.to_dict(); d["color_spaces"] = cs; out[k] = NetConfig.from_dict(d)
    return out


def fusion_configs(base: NetConfig) -> Dict[str, NetConfig]:
    return {f"fusion_{f}": NetConfig.from_dict({**base.to_dict(), "fusion": f}) for f in FUSION_ARMS}


def run_arms(arms: Dict[str, NetConfig], train_ds, test_ds, seeds: Sequence[int], tcfg: TrainConfig, val_ds=None, teacher=None) -> List[dict]:
    """Train each arm for each seed; return raw rows [arm, seed, miou, mf1, severity_mae, params]. ``teacher`` enables distillation."""
    rows = []
    for name, nc in arms.items():
        for s in seeds:
            tc = TrainConfig(**{**tcfg.__dict__, "seed": s, "use_normals_prior": nc.cartridge_prior})
            m = VPCorrosionNet(nc)
            train(m, train_ds, tc, val_ds, teacher=teacher)
            ev = evaluate(m, test_ds, tc)
            rows.append({"arm": name, "seed": s, **{k: ev["summary"][k] for k in ("miou", "mdice", "mf1", "mbalanced_accuracy")},
                         "severity_mae": ev["severity"].get("mae", float("nan")), "params": sum(p.numel() for p in m.parameters()),
                         "distilled": teacher is not None, "data": "SYNTHETIC" if isinstance(test_ds, SyntheticCorrosionDataset) else "REAL"})
    return rows
