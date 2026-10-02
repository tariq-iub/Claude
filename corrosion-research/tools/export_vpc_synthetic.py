"""Export vpc's synthetic polarimetric renders to the FolderCorrosionDataset layout (SYNTHETIC; wiring tests only).

Writes  <out>/images/<stem>.png, <out>/masks/<stem>.png (6-class corrosion-research taxonomy), <out>/pol/<stem>.npz
(stack (A,3,H,W) float16 linear analyzer images + angles) and <out>/groups.csv (stem, group_id).

vpc has 7 classes, corrosion-research has 6 (distinct oxide species + ignore). The default mapping is by visual/optical
analogue, NOT chemistry, and is an assumption to be revisited (--map overrides): background->ignore, healthy->healthy,
discoloration->cuprite, oxidation_dark (vpc 'cu2o_like')->cuprite, patina_green->atacamite, chloride_deposit->nantokite,
pitting (vpc 'cuo_like')->tenorite.

    python tools/export_vpc_synthetic.py --out data/vpc_synth --groups 24 --views 3
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common.polarization  # noqa: F401  (puts vpc on sys.path)
from vpc.experiments.synthetic import SynthConfig, generate_dataset  # noqa: E402

# vpc: background, healthy_metal, discoloration, oxidation_dark, patina_green, chloride_deposit, pitting
# here: healthy, cuprite, tenorite, nantokite, atacamite, ignore
DEFAULT_MAP = [5, 0, 1, 1, 4, 3, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", type=int, nargs=7, default=DEFAULT_MAP, help="target class id for each of the 7 vpc classes")
    a = ap.parse_args()
    from PIL import Image
    for d in ("images", "masks", "pol"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    lut = np.array(a.map, dtype=np.uint8)
    data = generate_dataset(a.groups, a.views, SynthConfig(size=a.size), seed=a.seed)
    rows = []
    for i, d in enumerate(data):
        stem = f"g{d['group_id']:03d}_v{i:04d}"
        rgb = (d["rgb"].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        mask = lut[d["mask"].numpy().astype(np.int64)]
        Image.fromarray(rgb).save(os.path.join(a.out, "images", stem + ".png"))
        Image.fromarray(mask).save(os.path.join(a.out, "masks", stem + ".png"))
        np.savez_compressed(os.path.join(a.out, "pol", stem + ".npz"), stack=d["stack"].numpy().astype(np.float16),
                            angles=d["angles"].numpy())
        rows.append((stem, d["group_id"]))
    with open(os.path.join(a.out, "groups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "group_id"])
        w.writerows(rows)
    print(f"wrote {len(rows)} samples from {a.groups} groups to {a.out}")


if __name__ == "__main__":
    main()
