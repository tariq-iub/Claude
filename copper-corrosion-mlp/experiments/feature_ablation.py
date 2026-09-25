"""Compare candidate feature representations under a fixed, small MLP and
fixed repeated-stratified-CV protocol. Selects the smallest feature set
that is statistically competitive with the best (task brief §20)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_labels, CANONICAL_CLASSES
from src.features import build_features, build_prototypes, FEATURE_SETS
from src.preprocessing import StandardScalerTrainOnly
from src.splits import repeated_stratified_folds
from src.trainer import train_model
from src.evaluator import predict_proba, compute_metrics


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])
    y = ds.label_ids
    n_classes = len(CANONICAL_CLASSES)

    folds = repeated_stratified_folds(y, n_splits=cfg["validation"]["folds"],
                                       n_repeats=cfg["validation"]["repeats"], seed=cfg["seed"])

    small_cfg = dict(cfg)
    small_cfg["model"] = {"kind": "plain", "hidden_dims": [32, 16], "activation": "gelu",
                           "dropout": 0.1, "batchnorm": True}
    small_cfg["training"] = dict(cfg["training"])
    small_cfg["training"]["epochs"] = 150
    small_cfg["training"]["patience"] = 20

    results = {}
    for fset in FEATURE_SETS:
        fold_f1 = []
        for train_idx, test_idx in folds:
            lab_train, lab_test = ds.lab[train_idx], ds.lab[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            prototypes = None
            if "deltaE" in fset:
                labels_train_names = ds.labels[train_idx]
                prototypes = build_prototypes(lab_train, labels_train_names, CANONICAL_CLASSES)

            X_train, _ = build_features(lab_train, fset, prototypes)
            X_test, _ = build_features(lab_test, fset, prototypes)

            # inner val split for early stopping (stratified subsplit of train)
            rng = np.random.default_rng(cfg["seed"])
            val_mask = np.zeros(len(train_idx), dtype=bool)
            for k in range(n_classes):
                idx_k = np.where(y_train == k)[0]
                n_val = max(1, int(round(0.2 * len(idx_k))))
                val_mask[rng.choice(idx_k, size=n_val, replace=False)] = True

            scaler = StandardScalerTrainOnly().fit(X_train[~val_mask])
            Xtr_s = scaler.transform(X_train[~val_mask])
            Xva_s = scaler.transform(X_train[val_mask])
            Xte_s = scaler.transform(X_test)

            model, hist, _ = train_model(Xtr_s, y_train[~val_mask], Xva_s, y_train[val_mask],
                                          X_train.shape[1], n_classes, small_cfg, seed=cfg["seed"])
            pred = predict_proba(model, Xte_s).argmax(axis=1)
            m = compute_metrics(y_test, pred, n_classes, CANONICAL_CLASSES)
            fold_f1.append(m["macro_f1"])
        results[fset] = {"macro_f1_mean": float(np.mean(fold_f1)), "macro_f1_std": float(np.std(fold_f1)),
                          "n_folds": len(fold_f1), "dim": X_train.shape[1]}
        print(f"{fset:20s} dim={X_train.shape[1]:2d}  macro-F1 = {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}")

    out_path = ROOT / "results" / "feature_ablation.json"
    out_path.parent.mkdir(exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    import csv
    with open(ROOT / "results" / "feature_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "dim", "macro_f1_mean", "macro_f1_std"])
        for k, v in results.items():
            w.writerow([k, v["dim"], v["macro_f1_mean"], v["macro_f1_std"]])
    return results


if __name__ == "__main__":
    run()
