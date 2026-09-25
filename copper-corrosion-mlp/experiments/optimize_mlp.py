"""Train and evaluate the final selected model (config-driven) on a
stratified 70/15/15 holdout, repeated across multiple seeds, plus
baselines, calibration, uncertainty thresholding, permutation importance
and complexity/latency reporting. Produces every results/ artifact for
the final proposed method."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_labels, load_settings, CANONICAL_CLASSES
from src.features import build_features, build_prototypes
from src.preprocessing import StandardScalerTrainOnly
from src.splits import train_val_test_split
from src.trainer import train_model
from src.evaluator import (predict_proba, predict_logits, compute_metrics, bootstrap_ci,
                            expected_calibration_error, brier_score)
from src.calibration import fit_temperature, apply_temperature
from src.uncertainty import tune_threshold, apply_abstention
from src.interpret import permutation_importance
from src.complexity import complexity_report
from src.baselines import nearest_prototype_ciede2000, nearest_centroid_euclidean, logistic_regression_baseline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])
    registry = load_settings(str(ROOT / cfg["data"]["settings_path"]))
    y = ds.label_ids
    n_classes = len(CANONICAL_CLASSES)
    fset = cfg["features"]["set"]

    seeds = cfg["seeds_for_repeat"]
    per_seed_metrics = []
    artifacts_for_first_seed = None

    for seed in seeds:
        train_idx, val_idx, test_idx = train_val_test_split(
            y, seed=seed, train_frac=cfg["validation"]["train_frac"], val_frac=cfg["validation"]["val_frac"])

        prototypes = None
        if "deltaE" in fset:
            prototypes = build_prototypes(ds.lab[train_idx], ds.labels[train_idx], CANONICAL_CLASSES)

        X_train, feat_names = build_features(ds.lab[train_idx], fset, prototypes)
        X_val, _ = build_features(ds.lab[val_idx], fset, prototypes)
        X_test, _ = build_features(ds.lab[test_idx], fset, prototypes)

        scaler = StandardScalerTrainOnly().fit(X_train)
        Xtr, Xva, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

        run_cfg = dict(cfg)
        model, hist, class_weights = train_model(Xtr, y[train_idx], Xva, y[val_idx],
                                                   Xtr.shape[1], n_classes, run_cfg, seed=seed)

        pred_test = predict_proba(model, Xte).argmax(axis=1)
        m_test = compute_metrics(y[test_idx], pred_test, n_classes, CANONICAL_CLASSES)
        m_train = compute_metrics(y[train_idx], predict_proba(model, Xtr).argmax(axis=1), n_classes, CANONICAL_CLASSES)
        m_val = compute_metrics(y[val_idx], predict_proba(model, Xva).argmax(axis=1), n_classes, CANONICAL_CLASSES)

        per_seed_metrics.append({
            "seed": seed, "test_accuracy": m_test["accuracy"], "test_balanced_accuracy": m_test["balanced_accuracy"],
            "test_macro_f1": m_test["macro_f1"], "test_weighted_f1": m_test["weighted_f1"],
            "train_accuracy": m_train["accuracy"], "val_accuracy": m_val["accuracy"],
            "best_epoch": hist.best_epoch, "stopping_reason": hist.stopping_reason,
        })

        if seed == seeds[0]:
            artifacts_for_first_seed = dict(
                model=model, scaler=scaler, feat_names=feat_names, prototypes=prototypes,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                Xtr=Xtr, Xva=Xva, Xte=Xte, m_train=m_train, m_val=m_val, m_test=m_test, hist=hist,
                class_weights=class_weights,
            )

    per_seed = np.array([[d["test_accuracy"], d["test_balanced_accuracy"], d["test_macro_f1"], d["test_weighted_f1"]]
                          for d in per_seed_metrics])
    stability = {
        "seeds": seeds,
        "test_accuracy": {"mean": float(per_seed[:, 0].mean()), "std": float(per_seed[:, 0].std())},
        "test_balanced_accuracy": {"mean": float(per_seed[:, 1].mean()), "std": float(per_seed[:, 1].std())},
        "test_macro_f1": {"mean": float(per_seed[:, 2].mean()), "std": float(per_seed[:, 2].std())},
        "test_weighted_f1": {"mean": float(per_seed[:, 3].mean()), "std": float(per_seed[:, 3].std())},
        "per_seed": per_seed_metrics,
    }
    print("Multi-seed test performance (mean +/- std over seeds", seeds, "):")
    for k in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_weighted_f1"]:
        print(f"  {k}: {stability[k]['mean']:.4f} +/- {stability[k]['std']:.4f}")

    # ---- Use first-seed artifacts for detailed reporting (calibration, uncertainty, etc.) ----
    a = artifacts_for_first_seed
    model, scaler, feat_names = a["model"], a["scaler"], a["feat_names"]
    train_idx, val_idx, test_idx = a["train_idx"], a["val_idx"], a["test_idx"]

    # Bootstrap CIs on test set (observation-level; no grouping metadata available -> documented limitation)
    y_test = y[test_idx]
    pred_test = predict_proba(model, a["Xte"]).argmax(axis=1)
    ci_acc = bootstrap_ci(y_test, pred_test, lambda yt, yp: accuracy_score(yt, yp), n_boot=1000, seed=cfg["seed"])
    ci_bacc = bootstrap_ci(y_test, pred_test, lambda yt, yp: balanced_accuracy_score(yt, yp), n_boot=1000, seed=cfg["seed"])
    ci_f1 = bootstrap_ci(y_test, pred_test, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
                          n_boot=1000, seed=cfg["seed"])

    # Calibration: temperature scaling fit on validation
    val_logits = predict_logits(model, a["Xva"])
    T = fit_temperature(val_logits, y[val_idx])
    test_logits = predict_logits(model, a["Xte"])
    proba_test_raw = predict_proba(model, a["Xte"])
    proba_test_cal = apply_temperature(test_logits, T)
    ece_raw = expected_calibration_error(y_test, proba_test_raw)
    ece_cal = expected_calibration_error(y_test, proba_test_cal)
    brier_raw = brier_score(y_test, proba_test_raw, n_classes)
    brier_cal = brier_score(y_test, proba_test_cal, n_classes)

    # Uncertainty / abstention, tuned on validation
    proba_val_cal = apply_temperature(predict_logits(model, a["Xva"]), T)
    thresh_info = tune_threshold(proba_val_cal, y[val_idx], min_coverage=cfg["uncertainty"]["min_coverage"])
    abst_pred = apply_abstention(proba_test_cal, thresh_info["tau"])
    coverage_test = float((abst_pred != -1).mean())
    retained_acc_test = float((abst_pred[abst_pred != -1] == y_test[abst_pred != -1]).mean()) if coverage_test > 0 else float("nan")

    # Interpretability
    perm_imp = permutation_importance(model, a["Xte"], y_test, feat_names, n_repeats=30, seed=cfg["seed"])

    # Complexity
    complexity = complexity_report(model, a["Xtr"].shape[1])

    # Baselines (same seed-0 split, fair comparison)
    lab_train, lab_test = ds.lab[train_idx], ds.lab[test_idx]
    y_train = y[train_idx]
    pred_nn_lab = nearest_prototype_ciede2000(lab_train, y_train, lab_test, n_classes)
    pred_nn_euclid = nearest_centroid_euclidean(a["Xtr"], y_train, a["Xte"], n_classes)
    pred_logreg, _ = logistic_regression_baseline(a["Xtr"], y_train, a["Xte"], seed=cfg["seed"])

    baselines = {
        "nearest_prototype_ciede2000": compute_metrics(y_test, pred_nn_lab, n_classes, CANONICAL_CLASSES),
        "nearest_centroid_euclidean": compute_metrics(y_test, pred_nn_euclid, n_classes, CANONICAL_CLASSES),
        "logistic_regression": compute_metrics(y_test, pred_logreg, n_classes, CANONICAL_CLASSES),
        "proposed_mlp": a["m_test"],
    }

    # ---- Save everything ----
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    json.dump(a["m_train"], open(out_dir / "metrics_train.json", "w"), indent=2)
    json.dump(a["m_val"], open(out_dir / "metrics_validation.json", "w"), indent=2)
    json.dump(a["m_test"], open(out_dir / "metrics_test.json", "w"), indent=2)
    json.dump(stability, open(out_dir / "multi_seed_stability.json", "w"), indent=2)
    json.dump({"accuracy": ci_acc, "balanced_accuracy": ci_bacc, "macro_f1": ci_f1},
               open(out_dir / "bootstrap_confidence_intervals.json", "w"), indent=2)
    json.dump({"ece_raw": ece_raw, "ece_calibrated": ece_cal, "brier_raw": brier_raw, "brier_calibrated": brier_cal,
               "temperature": T}, open(out_dir / "calibration_results.json", "w"), indent=2)
    json.dump({"threshold": thresh_info, "test_coverage": coverage_test, "test_retained_accuracy": retained_acc_test},
               open(out_dir / "uncertainty_results.json", "w"), indent=2)
    json.dump(perm_imp, open(out_dir / "permutation_importance.json", "w"), indent=2)
    json.dump(complexity, open(out_dir / "complexity_report.json", "w"), indent=2)
    json.dump(baselines, open(out_dir / "baseline_comparison.json", "w"), indent=2)
    json.dump(ds.quality_report, open(out_dir / "data_quality_report.json", "w"), indent=2)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    import torch
    torch.save(model.state_dict(), models_dir / "best_model.pt")
    json.dump({
        "feature_set": fset, "feature_names": feat_names, "hidden_dims": cfg["model"]["hidden_dims"],
        "model_kind": cfg["model"]["kind"], "activation": cfg["model"]["activation"],
        "dropout": cfg["model"]["dropout"], "batchnorm": cfg["model"]["batchnorm"],
        "scaler": scaler.to_dict(), "class_order": CANONICAL_CLASSES, "temperature": T,
        "abstention_threshold": thresh_info["tau"], "seed": seeds[0],
        "prototypes": prototypes.tolist() if prototypes is not None else None,
    }, open(models_dir / "preprocessing_and_config.json", "w"), indent=2)

    print("\nBaseline comparison (test macro-F1):")
    for k, v in baselines.items():
        print(f"  {k:28s} macro-F1={v['macro_f1']:.4f}  acc={v['accuracy']:.4f}")
    print(f"\nCalibration: ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f} (T={T:.3f})")
    print(f"Uncertainty: tau={thresh_info['tau']:.3f}  test coverage={coverage_test:.3f}  "
          f"retained accuracy={retained_acc_test:.3f}")
    print(f"Complexity: {complexity['trainable_parameters']} params, {complexity['fp32_size_kb']} KB, "
          f"latency(bs=1) mean={complexity['cpu_latency'][1]['mean_ms']:.4f} ms")

    return {"stability": stability, "baselines": baselines, "complexity": complexity}


if __name__ == "__main__":
    run()
