"""Staged ablation study (task brief §19): isolate the contribution of each
proposed change over the Prompt.md baseline, under one consistent repeated-
stratified-CV protocol so numbers are comparable across stages.

Stages:
 1. original MLP from Prompt.md (7-dim incl. delta, min-max norm, unweighted CE, plain 64-32-16)
 2. optimized architecture only (residual 32-32, z-score norm not yet -> still min-max, unweighted CE)
 3. + improved (train-only z-score) normalization
 4. + class balancing (inverse-frequency weighted CE)
 5. + final feature set (raw Lab, D=3, per feature_ablation.csv - drops delta/H/S/V)
 6. + residual block (already included from stage 2; stage kept for table completeness)
 7. + calibration (temperature scaling) - reported via ECE/Brier, not macro-F1
 8. final complete method (all of the above)
"""
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
from src.features import build_features, build_prototypes, lab_to_approx_rgb
from src import color_metrics as cm
from src.preprocessing import StandardScalerTrainOnly
from src.splits import repeated_stratified_folds
from src.trainer import train_model
from src.evaluator import predict_proba, compute_metrics, expected_calibration_error, predict_logits
from src.calibration import fit_temperature, apply_temperature


def minmax_fit_transform(Xtr, *others):
    mn, mx = Xtr.min(axis=0), Xtr.max(axis=0)
    rng_ = np.where(mx - mn < 1e-8, 1.0, mx - mn)
    f = lambda X: np.clip((X - mn) / rng_, 0.0, 1.0)
    return (f(Xtr),) + tuple(f(o) for o in others)


def inner_val_split(train_idx, y_train, n_classes, seed):
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(train_idx), dtype=bool)
    for k in range(n_classes):
        idx_k = np.where(y_train == k)[0]
        n_val = max(1, int(round(0.2 * len(idx_k))))
        val_mask[rng.choice(idx_k, size=n_val, replace=False)] = True
    return val_mask


def run_stage(name, ds, y, folds, n_classes, feature_fn, scaler_kind, model_cfg, loss_cfg, seed,
              report_calibration=False):
    fold_f1, fold_acc, fold_ece = [], [], []
    for train_idx, test_idx in folds:
        lab_train, lab_test = ds.lab[train_idx], ds.lab[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = feature_fn(lab_train, lab_test, ds.labels[train_idx])

        val_mask = inner_val_split(train_idx, y_train, n_classes, seed)
        Xtr_raw, Xva_raw = X_train[~val_mask], X_train[val_mask]

        if scaler_kind == "zscore":
            scaler = StandardScalerTrainOnly().fit(Xtr_raw)
            Xtr, Xva, Xte = scaler.transform(Xtr_raw), scaler.transform(Xva_raw), scaler.transform(X_test)
        else:
            Xtr, Xva, Xte = minmax_fit_transform(Xtr_raw, Xva_raw, X_test)

        run_cfg = {"model": model_cfg, "training": {
            "learning_rate": 0.003 if scaler_kind == "zscore" else 0.001,
            "weight_decay": 1e-4, "batch_size": min(32, len(Xtr)), "epochs": 200, "patience": 25,
            "scheduler": "plateau" if scaler_kind == "zscore" else "none"},
            "loss": loss_cfg}
        model, hist, _ = train_model(Xtr, y_train[~val_mask], Xva, y_train[val_mask],
                                      Xtr.shape[1], n_classes, run_cfg, seed=seed)
        pred = predict_proba(model, Xte).argmax(axis=1)
        m = compute_metrics(y_test, pred, n_classes, CANONICAL_CLASSES)
        fold_f1.append(m["macro_f1"])
        fold_acc.append(m["accuracy"])

        if report_calibration:
            val_logits = predict_logits(model, Xva)
            T = fit_temperature(val_logits, y_train[val_mask])
            test_logits = predict_logits(model, Xte)
            proba_cal = apply_temperature(test_logits, T)
            fold_ece.append(expected_calibration_error(y_test, proba_cal))

    result = {"macro_f1_mean": float(np.mean(fold_f1)), "macro_f1_std": float(np.std(fold_f1)),
              "accuracy_mean": float(np.mean(fold_acc)), "accuracy_std": float(np.std(fold_acc))}
    if report_calibration:
        result["ece_mean"] = float(np.mean(fold_ece))
    print(f"{name:45s} acc={result['accuracy_mean']:.4f}  macro-F1={result['macro_f1_mean']:.4f} "
          f"+/- {result['macro_f1_std']:.4f}" + (f"  ECE={result.get('ece_mean', float('nan')):.4f}" if report_calibration else ""))
    return result


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])
    y = ds.label_ids
    n_classes = len(CANONICAL_CLASSES)
    seed = cfg["seed"]
    folds = repeated_stratified_folds(y, n_splits=cfg["validation"]["folds"],
                                       n_repeats=cfg["validation"]["repeats"], seed=seed)

    def feat_promptmd(lab_train, lab_test, labels_train):
        healthy_proto = build_prototypes(lab_train, labels_train, ["Healthy"])
        def f(lab):
            rgb = lab_to_approx_rgb(lab)
            hsv = cm.rgb_to_hsv(rgb)
            delta = cm.ciede2000(lab, healthy_proto).min(axis=1, keepdims=True)
            return np.concatenate([lab, hsv, delta], axis=1)
        return f(lab_train), f(lab_test)

    def feat_lab(lab_train, lab_test, labels_train):
        Xtr, _ = build_features(lab_train, "lab")
        Xte, _ = build_features(lab_test, "lab")
        return Xtr, Xte

    plain_arch = {"kind": "plain", "hidden_dims": [64, 32, 16], "activation": "relu", "dropout": 0.15, "batchnorm": False}
    resid_arch = {"kind": "residual", "hidden_dims": [32, 32], "activation": "gelu", "dropout": 0.2, "batchnorm": True}
    unweighted = {"type": "weighted_ce", "class_weights": "none"}
    weighted = {"type": "weighted_ce", "class_weights": "inverse-frequency"}

    results = {}
    results["1_promptmd_baseline"] = run_stage(
        "1. Prompt.md baseline (7-dim, minmax, plain, unweighted CE)",
        ds, y, folds, n_classes, feat_promptmd, "minmax", plain_arch, unweighted, seed)

    results["2_optimized_architecture_only"] = run_stage(
        "2. + optimized architecture (residual 32-32), still minmax/unweighted",
        ds, y, folds, n_classes, feat_promptmd, "minmax", resid_arch, unweighted, seed)

    results["3_plus_improved_normalization"] = run_stage(
        "3. + train-only z-score normalization",
        ds, y, folds, n_classes, feat_promptmd, "zscore", resid_arch, unweighted, seed)

    results["4_plus_class_balancing"] = run_stage(
        "4. + inverse-frequency class-weighted CE",
        ds, y, folds, n_classes, feat_promptmd, "zscore", resid_arch, weighted, seed)

    results["5_final_feature_set_lab_only_weighted"] = run_stage(
        "5. + final feature set (raw Lab, D=3; drop delta/HSV per feature ablation), still weighted CE",
        ds, y, folds, n_classes, feat_lab, "zscore", resid_arch, weighted, seed)

    results["6_final_feature_set_unweighted"] = run_stage(
        "6. Lab features + z-score + residual, UNWEIGHTED CE (best point estimate; class weighting reverted per stage 4 negative result)",
        ds, y, folds, n_classes, feat_lab, "zscore", resid_arch, unweighted, seed)

    results["8_final_complete_with_calibration"] = run_stage(
        "8. Final complete method (6 + temperature scaling, ECE reported)",
        ds, y, folds, n_classes, feat_lab, "zscore", resid_arch, unweighted, seed, report_calibration=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    json.dump(results, open(out_dir / "ablation_results.json", "w"), indent=2)
    with open(out_dir / "ablation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "ece_mean"])
        for k, v in results.items():
            w.writerow([k, v["accuracy_mean"], v["accuracy_std"], v["macro_f1_mean"], v["macro_f1_std"], v.get("ece_mean", "")])
    return results


if __name__ == "__main__":
    run()
