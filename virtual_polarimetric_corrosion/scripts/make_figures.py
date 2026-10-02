#!/usr/bin/env python3
"""Generate publication figures. Usage: python scripts/make_figures.py [--only 1 3 4] [--results results] [--out figures]
Figures that need experiment results are skipped (with a message) when those results do not exist."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vpc.visualization import fig01_architecture
from vpc.visualization.figures import REGISTRY

ap = argparse.ArgumentParser()
ap.add_argument("--only", type=int, nargs="*")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
ap.add_argument("--tables", default="tables")
a = ap.parse_args()
todo = a.only or [1] + sorted(REGISTRY)
for n in todo:
    paths = fig01_architecture.make(a.out) if n == 1 else REGISTRY[n](outdir=a.out, results=a.results, tables=a.tables)
    print(f"Figure {n}:", paths if paths else "skipped")
