"""
End-to-end benchmark runner: trains every proposed architecture and the
tiny-U-Net deep baseline on the synthetic corrosion dataset, evaluates the
classical (non-learned) baselines on the same validation split, and reports
a single comparison table (mIoU, Dice, F1, boundary IoU/F1, params, speed)
across the fixed set of seeds requested by the research protocol.

Usage:
    python benchmarks/run_benchmark.py --epochs 10 --seeds 0 1 2 --cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from common.dataset import SyntheticCorrosionDataset, NUM_CLASSES, set_all_seeds
from common.engine import train, evaluate, get_device, make_loaders, count_params, benchmark_speed
from common.metrics import summarize
from benchmarks.baselines import (
    build_tiny_unet, kmeans_segment, gmm_segment, slic_classify,
    lab_deltae_classify, lut3d_classify, voxel5d_classify, estimate_class_centroids,
)

ARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "architectures")
ARCHS = [
    ("01_pmd_net", "PMD-Net"),
    ("02_cpen", "CPEN"),
    ("03_hypercornet", "HyperCorNet"),
    ("04_corronca", "CorroNCA"),
    ("05_corro_ssm", "Corro-SSM"),
    ("06_eb_toposeg", "EB-TopoSeg"),
    ("07_incf", "INCF"),
    ("08_momer", "MoMER"),
]


def load_build_model(folder: str):
    path = os.path.join(ARCH_DIR, folder, "model.py")
    spec = importlib.util.spec_from_file_location(f"{folder}_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


def run_classical_baselines(height, width, seed):
    val_ds = SyntheticCorrosionDataset(length=20, height=height, width=width, seed=seed + 999)
    ref_ds = SyntheticCorrosionDataset(length=10, height=height, width=width, seed=seed)  # small "reference" set for centroid estimation

    ref_imgs = [ref_ds[i][0].permute(1, 2, 0).numpy() for i in range(len(ref_ds))]
    ref_masks = [ref_ds[i][1].numpy() for i in range(len(ref_ds))]
    lab_centroids = estimate_class_centroids(ref_imgs, ref_masks, NUM_CLASSES, space="lab")
    rgb_centroids = estimate_class_centroids(ref_imgs, ref_masks, NUM_CLASSES, space="rgb")
    labxy_centroids = estimate_class_centroids(ref_imgs, ref_masks, NUM_CLASSES, space="labxy")

    results = {}
    methods = {
        "KMeans": lambda img: kmeans_segment(img),
        "GMM": lambda img: gmm_segment(img),
        "SLIC+LabCentroid": lambda img: slic_classify(img, lab_centroids),
        "Lab+DeltaE00": lambda img: lab_deltae_classify(img, lab_centroids),
        "3D-RGB-LUT": lambda img: lut3d_classify(img, rgb_centroids),
        "5D-Labxy-Voxel": lambda img: voxel5d_classify(img, labxy_centroids),
    }
    for name, fn in methods.items():
        preds, targets, probs = [], [], []
        for i in range(len(val_ds)):
            img_t, mask_t = val_ds[i]
            img = img_t.permute(1, 2, 0).numpy()
            pred = fn(img)
            preds.append(torch.from_numpy(pred).long())
            targets.append(mask_t)
            onehot = torch.nn.functional.one_hot(torch.from_numpy(pred).long().clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(2, 0, 1).float()
            probs.append(onehot)
        preds_t = torch.stack(preds)
        targets_t = torch.stack(targets)
        probs_t = torch.stack(probs)
        m = summarize(preds_t, targets_t, probs_t, NUM_CLASSES)
        results[name] = {"miou": m["miou"], "mdice": m["mdice"], "mf1": m["mf1"], "boundary_iou": m["boundary_iou"]}
        print(f"  [classical] {name}: mIoU={m['miou']:.4f} F1={m['mf1']:.4f}")
    return results


def run_learned_models(epochs, height, width, seeds, out_root, prefer_cpu):
    all_results = {}
    model_specs = [(folder, name, lambda f=folder: load_build_model(f)()) for folder, name in ARCHS]
    model_specs.append(("tiny_unet", "Tiny-U-Net (baseline)", build_tiny_unet))

    for folder, name, model_fn in model_specs:
        seed_metrics = []
        for seed in seeds:
            print(f"[{name}] seed={seed} training...")
            out_dir = os.path.join(out_root, folder, f"seed{seed}")
            model, history = train(
                model_fn=model_fn, out_dir=out_dir, epochs=epochs,
                batch_size=4, height=height, width=width, seed=seed, prefer_cpu=prefer_cpu,
            )
            device = get_device(prefer_cpu)
            _, val_loader = make_loaders(height, width, batch_size=4, seed=seed)
            metrics = evaluate(model, val_loader, device)
            speed = benchmark_speed(model, device, height=height, width=width)
            seed_metrics.append({**metrics, **speed, "params": count_params(model)})
            print(f"  seed {seed}: mIoU={metrics['miou']:.4f} F1={metrics['mf1']:.4f} latency={speed['latency_ms']:.1f}ms")

        def agg(key):
            vals = [m[key] for m in seed_metrics]
            return float(np.mean(vals)), float(np.std(vals))

        all_results[name] = {
            "miou": agg("miou"), "mdice": agg("mdice"), "mf1": agg("mf1"),
            "boundary_iou": agg("boundary_iou"), "hausdorff": agg("hausdorff"), "ece": agg("ece"),
            "latency_ms": agg("latency_ms"), "params": seed_metrics[0]["params"],
        }
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--skip-classical", action="store_true")
    args = ap.parse_args()

    report = {}
    if not args.skip_classical:
        print("=== Classical baselines ===")
        report["classical"] = run_classical_baselines(args.height, args.width, args.seeds[0])

    print("=== Learned architectures + Tiny-U-Net baseline ===")
    report["learned"] = run_learned_models(args.epochs, args.height, args.width, args.seeds, args.out, args.cpu)

    os.makedirs(args.out, exist_ok=True)
    report_path = os.path.join(args.out, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {report_path}")

    print("\n=== Summary (mean +/- std over seeds) ===")
    print(f"{'Model':<24}{'mIoU':<18}{'Dice':<18}{'F1':<18}{'Params':<10}")
    for name, m in report.get("learned", {}).items():
        miou = f"{m['miou'][0]:.3f}+/-{m['miou'][1]:.3f}"
        dice = f"{m['mdice'][0]:.3f}+/-{m['mdice'][1]:.3f}"
        f1 = f"{m['mf1'][0]:.3f}+/-{m['mf1'][1]:.3f}"
        print(f"{name:<24}{miou:<18}{dice:<18}{f1:<18}{m['params']:<10}")
    for name, m in report.get("classical", {}).items():
        print(f"{name:<24}{m['miou']:<18.3f}{m['mdice']:<18.3f}{m['mf1']:<18.3f}{'n/a':<10}")


if __name__ == "__main__":
    main()
