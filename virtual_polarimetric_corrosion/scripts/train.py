#!/usr/bin/env python3
"""Train VP-CorrosionNet (or a baseline) from a config. Usage: python scripts/train.py --config configs/default.yaml [--model vp|unet|mobile_rgb] [--seed 0]"""
import _common  # noqa
import argparse, os, torch
from vpc.experiments.pipeline import build_splits, load_config, net_config, train_config
from vpc.experiments.train import evaluate, save_run, train
from vpc.models.baseline import build_baseline
from vpc.models.vp_corrosion_net import VPCorrosionNet

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--model", default="vp"); ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--out", default="results/runs")
a = ap.parse_args()
cfg = load_config(a.config)
tr, va, te, info = build_splits(cfg)
tc = train_config(cfg, a.seed); nc = net_config(cfg); tc.use_normals_prior = nc.cartridge_prior
model = VPCorrosionNet(nc) if a.model == "vp" else build_baseline(a.model)
hist = train(model, tr, tc, va)
ev = evaluate(model, te, tc)
name = f"{a.model}_seed{tc.seed}"
os.makedirs(a.out, exist_ok=True)
torch.save(model.state_dict(), os.path.join(a.out, name + ".pt"))
save_run(os.path.join(a.out, name + ".json"), {"data_info": info, "train": hist, "test_summary": ev["summary"], "severity": ev["severity"], "net": nc.to_dict()})
print(info, ev["summary"])
