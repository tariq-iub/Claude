#!/usr/bin/env python3
"""Verify no cartridge group appears in more than one split (manifest CSV)."""
import _common  # noqa
import argparse, pandas as pd
from vpc.experiments.splits import assert_no_group_leakage
ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--split-col", default="split")
a = ap.parse_args()
df = pd.read_csv(a.manifest)
assert_no_group_leakage({s: g["group_id"].values for s, g in df.groupby(a.split_col)})
print("OK: no group leakage across", list(df[a.split_col].unique()))
