"""Compare candidate MLP architectures (width/depth/activation/residual)
under repeated-stratified-CV, fixed feature set (from feature_ablation.py).
Selects the smallest model statistically competitive with the best."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_labels, CANONICAL_CLASSES
from src.features import build_features
from src.preprocessing import StandardScalerTrainOnly
from src.splits import repeated_stratified_folds
from src.trainer import train_model
from src.evaluator import predict_proba, compute_metrics
from src.model import build_model, count_parameters

CANDIDATES = {
    "tiny_16":        {"kind": "plain",    "hidden_dims": [16],        "activation": "relu",  "dropout": 0.0,  "batchnorm": False},
    "plain_32_16":    {"kind": "plain",    "hidden_dims": [32, 16],    "activation": "gelu",  "dropout": 0.1,  "batchnorm": True},
    "plain_64_32":    {"kind": "plain",    "hidden_dims": [64, 32],    "activation": "relu",  "dropout": 0.15, "batchnorm": True},
    "plain_64_32_16": {"kind": "plain",    "hidden_dims": [64, 32, 16],"activation": "relu",  "dropout": 0.15, "batchnorm": True},
    "residual_32_32": {"kind": "residual", "hidden_dims": [32, 32],    "activation": "gelu",  "dropout": 0.2,  "batchnorm": True},
    "residual_64_64": {"kind": "residual", "hidden_dims": [64, 64],    "activation": "silu",  "dropout": 0.2,  "batchnorm": True},
}


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])
    y = ds.label_ids
    n_classes = len(CANONICAL_CLASSES)
    fset = cfg["features"]["set"]

    folds = repeated_stratified_folds(y, n_splits=cfg["validation"]["folds"],
                                       n_repeats=cfg["validation"]["repeats"], seed=cfg["seed"])

    results = {}
    for name, arch_cfg in CANDIDATES.items():
        run_cfg = dict(cfg)
        run_cfg["model"] = arch_cfg
        run_cfg["training"] = dict(cfg["training"])
        run_cfg["training"]["epochs"] = 200
        run_cfg["training"]["patience"] = 25

        fold_f1, n_params = [], None
        for train_idx, test_idx in folds:
            lab_train, lab_test = ds.lab[train_idx], ds.lab[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, feat_names = build_features(lab_train, fset)
            X_test, _ = build_features(lab_test, fset)

            rng = np.random.default_rng(cfg["seed"])
            val_mask = np.zeros(len(train_idx), dtype=bool)
            for k in range(n_classes):
                idx_k = np.where(y_train == k)[0]
                n_val = max(1, int(round(0.2 * len(idx_k))))
                val_mask[rng.choice(idx_k, size=n_val, replace=False)] = True

            scaler = StandardScalerTrainOnly().fit(X_train[~val_mask])
            Xtr_s, Xva_s, Xte_s = scaler.transform(X_train[~val_mask]), scaler.transform(X_train[val_mask]), scaler.transform(X_test)

            model, hist, _ = train_model(Xtr_s, y_train[~val_mask], Xva_s, y_train[val_mask],
                                          X_train.shape[1], n_classes, run_cfg, seed=cfg["seed"])
            n_params = count_parameters(model)
            pred = predict_proba(model, Xte_s).argmax(axis=1)
            m = compute_metrics(y_test, pred, n_classes, CANONICAL_CLASSES)
            fold_f1.append(m["macro_f1"])

        results[name] = {"macro_f1_mean": float(np.mean(fold_f1)), "macro_f1_std": float(np.std(fold_f1)),
                          "n_params": n_params, "config": arch_cfg}
        print(f"{name:16s} params={n_params:4d}  macro-F1 = {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    json.dump(results, open(out_dir / "architecture_ablation.json", "w"), indent=2)
    with open(out_dir / "architecture_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "n_params", "macro_f1_mean", "macro_f1_std"])
        for k, v in results.items():
            w.writerow([k, v["n_params"], v["macro_f1_mean"], v["macro_f1_std"]])
    return results


if __name__ == "__main__":
    run()
