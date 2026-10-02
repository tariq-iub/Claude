"""Wiring benchmark for polarization input arms on SYNTHETIC vpc data (validates plumbing, NOT science).

Because the synthetic generator itself places class signal in the polarization channels, a gain here only shows the
pipeline can use the channels; the class-independent control arm (``*_shuffled``) must stay at the colour-only level.
Group-level split (cartridge ids from groups.csv); Tiny-U-Net inside PolFusionWrapper so every arm shares one code path.

    python tools/export_vpc_synthetic.py --out data/vpc_synth --groups 24 --views 3
    python benchmarks/run_pol_wiring.py --data data/vpc_synth --epochs 8 --seeds 0 1 --cpu
"""
import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.baselines import TinyUNet
from common.dataset import FolderCorrosionDataset, NUM_CLASSES
from common.engine import train, evaluate, get_device, count_params
from common.polarization import PolFusionWrapper, pol_channels

ARMS = [("A0_rgb", "none", False), ("A2_dolp", "dolp", False), ("A3_stokes", "stokes", False),
        ("A4_stokes_aolp", "stokes_aolp", False), ("A5_stack", "stack", False),
        ("A7_dolp_shuffled", "dolp", True), ("A7_stokes_shuffled", "stokes", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="runs/pol_wiring")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.data, "groups.csv"))))
    stems = sorted(r["stem"] for r in rows)
    gid = {r["stem"]: int(r["group_id"]) for r in rows}
    groups = sorted(set(gid.values()))
    val_groups = set(groups[int(0.7 * len(groups)):])
    all_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(a.data, "images")))
    tr = [i for i, s in enumerate(all_stems) if gid[s] not in val_groups]
    va = [i for i, s in enumerate(all_stems) if gid[s] in val_groups]
    assert not ({gid[all_stems[i]] for i in tr} & {gid[all_stems[i]] for i in va}), "group leakage"

    n_ang = len(np.load(os.path.join(a.data, 'pol', all_stems[0] + '.npz'))['angles'])
    device = get_device(a.cpu)
    results = []
    for name, mode, shuffle in ARMS:
        for seed in a.seeds:
            p = pol_channels(mode, n_ang)
            tr_ds = FolderCorrosionDataset(a.data, pol_mode=mode, augment=True, pol_shuffle=shuffle, seed=seed, indices=tr)
            va_ds = FolderCorrosionDataset(a.data, pol_mode=mode, pol_shuffle=shuffle, seed=seed + 999, indices=va)
            loaders = (DataLoader(tr_ds, a.batch, shuffle=True), DataLoader(va_ds, a.batch))
            torch.manual_seed(seed)
            model, _ = train(lambda: PolFusionWrapper(TinyUNet(), NUM_CLASSES, p), os.path.join(a.out, f"{name}_s{seed}"),
                             epochs=a.epochs, seed=seed, prefer_cpu=a.cpu, amp=False, loaders=loaders)
            m = evaluate(model, loaders[1], device)
            results.append({"arm": name, "seed": seed, "miou": float(m["miou"]), "mf1": float(m["mf1"]),
                            "params": count_params(model), "data": "SYNTHETIC"})
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("\narm, mean mIoU, mean mF1 (SYNTHETIC; wiring check only)")
    for name, _, _ in ARMS:
        r = [x for x in results if x["arm"] == name]
        print(f"{name:20s} {np.mean([x['miou'] for x in r]):.4f} {np.mean([x['mf1'] for x in r]):.4f}  params={r[0]['params']}")


if __name__ == "__main__":
    main()
