"""Metrics, confusion matrices, bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score,
)


def predict_logits(model, X, device="cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(x)
    return logits.cpu().numpy()


def predict_proba(model, X, device="cpu") -> np.ndarray:
    logits = predict_logits(model, X, device=device)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, class_names: list) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    specificity = []
    for k in range(n_classes):
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    per_class = {
        class_names[k]: {
            "precision": float(precision[k]), "recall": float(recall[k]),
            "specificity": float(specificity[k]), "f1": float(f1[k]), "support": int(support[k]),
        } for k in range(n_classes)
    }
    return {
        "accuracy": float(acc), "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).round(4).tolist(),
        "class_order": class_names,
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42,
                  alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(stats.mean()), "std": float(stats.std()), "ci_lo": float(lo), "ci_hi": float(hi)}


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
