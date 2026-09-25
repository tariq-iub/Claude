"""Train EB-TopoSeg on the synthetic corrosion benchmark.

Usage:
    python train.py --epochs 15 --cpu
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.dirname(__file__))

from common.engine import train
from model import build_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "seed0"))
    args = ap.parse_args()

    train(
        model_fn=build_model,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        height=args.height,
        width=args.width,
        seed=args.seed,
        prefer_cpu=args.cpu,
    )
