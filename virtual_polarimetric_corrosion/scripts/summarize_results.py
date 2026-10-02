#!/usr/bin/env python3
"""Summarise raw per-seed result CSVs (results/*.csv) into mean/sd/CI tables in tables/ (does nothing for missing files)."""
import _common  # noqa
import os, pandas as pd
from vpc.experiments.tables import summarize

JOBS = {"ablation": (["arm"], ["miou", "mdice", "mf1", "mbalanced_accuracy", "severity_mae"]),
        "overall_segmentation": (["method"], ["miou", "mdice", "mf1", "mbalanced_accuracy"]),
        "robustness": (["method", "perturbation", "level"], ["miou", "mf1"]),
        "physical_vs_virtual": (["config"], ["miou", "mf1", "severity_mae", "pol_dolp_mae", "pol_aolp_wrapped_err_deg"])}
os.makedirs("tables", exist_ok=True)
for name, (g, m) in JOBS.items():
    p = f"results/{name}.csv"
    if not os.path.exists(p):
        print("skip", p); continue
    out = summarize(pd.read_csv(p), g, m)
    out.to_csv(f"tables/{name}_summary.csv", index=False); print("wrote", f"tables/{name}_summary.csv", len(out))
