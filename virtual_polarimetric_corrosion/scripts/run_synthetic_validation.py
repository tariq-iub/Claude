#!/usr/bin/env python3
"""Validate virtual-physics modules against synthetic ground truth (SYNTHETIC ONLY). Writes results/synthetic_validation.csv."""
import _common  # noqa
import argparse, os
import pandas as pd
from vpc.experiments.pipeline import synthetic_validation

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); ap.add_argument("--out", default="results/synthetic_validation.csv")
a = ap.parse_args()
rows = synthetic_validation(a.n)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(df.to_string(index=False))
