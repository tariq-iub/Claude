#!/usr/bin/env python3
"""Compute measured Stokes/DoLP/AoLP from a folder of analyzer images. Usage: process_hardware_stack.py FOLDER OUT.npz --angles 0 45 90 135 [--dark d.png --flat f.png]"""
import _common  # noqa
import argparse
from vpc.experiments.hardware_io import process_folder
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("out"); ap.add_argument("--angles", type=float, nargs="+", default=[0, 45, 90, 135])
ap.add_argument("--pattern", default="I{angle}.png"); ap.add_argument("--dark"); ap.add_argument("--flat"); ap.add_argument("--zero-offset", type=float, default=0.0)
a = ap.parse_args()
r = process_folder(a.folder, a.angles, a.out, pattern=a.pattern, dark=a.dark, flat=a.flat, zero_offset_deg=a.zero_offset)
print({k: v.shape for k, v in r.items()}, "fit RMSE (mean):", float(r["fit_rmse"].mean()))
