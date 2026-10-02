#!/usr/bin/env python3
"""Hardware-to-software gap experiment (A physical teacher / B RGB virtual / C distilled). Needs data with analyzer stacks."""
import _common  # noqa
import argparse
from vpc.experiments.hardware_gap import run_hardware_gap
from vpc.experiments.pipeline import build_splits, load_config, net_config, train_config
from vpc.experiments.tables import append_rows

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/hardware_teacher.yaml")
a = ap.parse_args()
cfg = load_config(a.config)
tr, va, te, info = build_splits(cfg)
rows = run_hardware_gap(tr, te, train_config(cfg), net_config(cfg), cfg.get("seeds", [0]), va)
for r in rows:
    r["data_origin"] = info["origin"]
print(append_rows("physical_vs_virtual", [{k: v for k, v in r.items() if k in __import__("vpc.experiments.tables", fromlist=["SCHEMAS"]).SCHEMAS["physical_vs_virtual"] or k == "config"} for r in rows], "results"))
