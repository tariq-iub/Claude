#!/usr/bin/env python3
"""Run ablation arms x seeds; append RAW per-seed rows to results/ablation.csv. Usage: python scripts/run_ablation.py --config configs/ablation.yaml"""
import _common  # noqa
import argparse
from vpc.experiments import ablation as A
from vpc.experiments.pipeline import build_splits, load_config, net_config, train_config
from vpc.experiments.tables import append_rows

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/ablation.yaml"); ap.add_argument("--arms", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
tr, va, te, info = build_splits(cfg)
kind = a.arms or cfg.get("arms", "cumulative")
full = net_config(cfg)
arms = {"cumulative": A.cumulative_configs, "leave_one_out": lambda: A.leave_one_out_configs(full),
        "color_space": lambda: A.color_space_configs(full), "fusion": lambda: A.fusion_configs(full)}[kind]()
rows = A.run_arms(arms, tr, te, cfg.get("seeds", [0, 1, 2]), train_config(cfg), va)
print(append_rows("ablation", rows, "results"))
