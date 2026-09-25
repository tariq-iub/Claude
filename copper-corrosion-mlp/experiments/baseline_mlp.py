"""Reproduce the MLP originally proposed in Prompt.md (7-dim
[L,a,b,H,S,V,delta], Dense(64)-Dense(32)-Dense(16)-Dense(5), plain
cross-entropy, no class weighting) as the ablation study's baseline
(task brief §19, item 1).

Deviations forced by the data actually available:
- Prompt.md specifies a `delta` input column that does not exist in
  labels.xlsx and is not defined anywhere in settings.xlsx. We substitute
  DeltaMode="ciede2000" (minimum Delta-E00 to training-derived Healthy
  prototypes), which is the one delta definition Prompt.md itself allows
  and is the only one computable from the available data.
- Prompt.md's 5-class taxonomy omits chloride vs. oxide distinctions
  consistently with settings.xlsx/labels.xlsx (H/Cu2O/CuO/CuCl/CuCl2), so
  no class remapping is needed.
"""
from __future__ import annotations

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
from src.splits import train_val_test_split
from src.trainer import train_model
from src.evaluator import predict_proba, compute_metrics


def build_prompt_md_features(lab, prototypes_healthy):
    rgb = lab_to_approx_rgb(lab)
    hsv = cm.rgb_to_hsv(rgb)
    delta = cm.ciede2000(lab, prototypes_healthy).min(axis=1, keepdims=True)
    X = np.concatenate([lab, hsv, delta], axis=1)
    return X, ["L", "a", "b", "H", "S", "V", "delta_ciede2000_to_healthy"]


def run(cfg_path=str(ROOT / "configs/mlp_config.yaml")):
    cfg = yaml.safe_load(open(cfg_path))
    ds = load_labels(str(ROOT / cfg["data"]["labels_path"]), cfg["data"]["invalid_row_policy"])
    y = ds.label_ids
    n_classes = len(CANONICAL_CLASSES)

    train_idx, val_idx, test_idx = train_val_test_split(y, seed=cfg["seed"],
                                                          train_frac=cfg["validation"]["train_frac"],
                                                          val_frac=cfg["validation"]["val_frac"])

    healthy_proto = build_prototypes(ds.lab[train_idx], ds.labels[train_idx], ["Healthy"])
    X_train, feat_names = build_prompt_md_features(ds.lab[train_idx], healthy_proto)
    X_val, _ = build_prompt_md_features(ds.lab[val_idx], healthy_proto)
    X_test, _ = build_prompt_md_features(ds.lab[test_idx], healthy_proto)

    # Prompt.md: min-max normalization (not z-score), fit on train only
    mn, mx = X_train.min(axis=0), X_train.max(axis=0)
    rng_ = np.where(mx - mn < 1e-8, 1.0, mx - mn)
    scale = lambda X: np.clip((X - mn) / rng_, 0.0, 1.0)
    Xtr, Xva, Xte = scale(X_train), scale(X_val), scale(X_test)

    prompt_cfg = dict(cfg)
    prompt_cfg["model"] = {"kind": "plain", "hidden_dims": [64, 32, 16], "activation": "relu",
                            "dropout": 0.15, "batchnorm": False}
    prompt_cfg["training"] = {"learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": min(32, len(Xtr)),
                               "epochs": 200, "patience": 10, "scheduler": "none"}
    prompt_cfg["loss"] = {"type": "weighted_ce", "class_weights": "none"}

    model, hist, _ = train_model(Xtr, y[train_idx], Xva, y[val_idx], Xtr.shape[1], n_classes,
                                  prompt_cfg, seed=cfg["seed"])
    pred_test = predict_proba(model, Xte).argmax(axis=1)
    metrics = compute_metrics(y[test_idx], pred_test, n_classes, CANONICAL_CLASSES)
    print("Prompt.md-equivalent MLP (7-dim, unweighted CE, no residual, no calibration):")
    print(f"  test accuracy = {metrics['accuracy']:.4f}  macro-F1 = {metrics['macro_f1']:.4f}"
          f"  balanced_accuracy = {metrics['balanced_accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    run()
