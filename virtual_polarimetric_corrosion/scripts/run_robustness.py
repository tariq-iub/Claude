#!/usr/bin/env python3
"""Degradation curves for a trained model over photometric perturbations. Appends RAW rows to results/robustness.csv."""
import _common  # noqa
import argparse, torch
from vpc.experiments.datasets import collate
from vpc.experiments.pipeline import build_splits, load_config, net_config, train_config
from vpc.experiments.robustness import PHOTOMETRIC, degradation_curve
from vpc.experiments.tables import append_rows
from vpc.experiments.train import select_device, train
from vpc.models.vp_corrosion_net import VPCorrosionNet
from vpc.models.baseline import build_baseline

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--models", nargs="*", default=["vp", "mobile_rgb"])
a = ap.parse_args()
cfg = load_config(a.config)
tr, va, te, info = build_splits(cfg)
b = collate([te[i] for i in range(len(te))])
rows = []
for seed in cfg.get("seeds", [0]):
    tc = train_config(cfg, seed)
    for name in a.models:
        m = VPCorrosionNet(net_config(cfg)) if name == "vp" else build_baseline(name)
        train(m, tr, tc, va); dev = select_device(tc.device); m.to(dev).eval()
        pred = lambda x: m(x.to(dev))["seg_logits"].argmax(1).cpu()
        for p in PHOTOMETRIC:
            for r in degradation_curve(pred, b["rgb"], b["mask"], p):
                rows.append({"method": name, "seed": seed, "perturbation": p, "level": r["level"], "miou": r["miou"], "mf1": r["mf1"], "data_origin": info["origin"]})
print(append_rows("robustness", rows, "results"))
