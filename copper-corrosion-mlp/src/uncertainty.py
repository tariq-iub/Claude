"""Confidence-based abstention: max_k P(y=k|x) < tau -> 'Uncertain'.

tau is tuned on the validation partition to maximize accuracy on the
retained (non-abstained) predictions subject to a minimum coverage
constraint, never on test data.
"""
from __future__ import annotations

import numpy as np


def tune_threshold(proba_val: np.ndarray, y_val: np.ndarray, min_coverage: float = 0.7,
                    grid: np.ndarray | None = None) -> dict:
    if grid is None:
        grid = np.linspace(0.3, 0.95, 27)
    conf = proba_val.max(axis=1)
    pred = proba_val.argmax(axis=1)
    correct = (pred == y_val)
    best = {"tau": 0.0, "retained_accuracy": float(correct.mean()), "coverage": 1.0}
    for tau in grid:
        keep = conf >= tau
        coverage = keep.mean()
        if coverage < min_coverage or keep.sum() == 0:
            continue
        retained_acc = correct[keep].mean()
        if retained_acc >= best["retained_accuracy"]:
            best = {"tau": float(tau), "retained_accuracy": float(retained_acc), "coverage": float(coverage)}
    return best


def apply_abstention(proba: np.ndarray, tau: float) -> np.ndarray:
    """Returns predicted class id, or -1 for 'Uncertain'."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    return np.where(conf >= tau, pred, -1)


def min_distance_to_prototypes(dE00_matrix: np.ndarray) -> np.ndarray:
    """dE00_matrix: (N,K) Delta-E00 to each class prototype -> (N,) min distance,
    a simple out-of-distribution / unusual-color signal independent of the
    classifier's own softmax confidence."""
    return dE00_matrix.min(axis=1)
