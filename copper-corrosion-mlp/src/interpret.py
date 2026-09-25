"""Permutation feature importance (model-agnostic, no SHAP dependency needed
for a 9-11 dim tabular input)."""
from __future__ import annotations

import numpy as np

from .evaluator import predict_proba
from sklearn.metrics import f1_score


def permutation_importance(model, X, y, feature_names, n_repeats=30, seed=42, device="cpu") -> dict:
    rng = np.random.default_rng(seed)
    base_pred = predict_proba(model, X, device=device).argmax(axis=1)
    base_score = f1_score(y, base_pred, average="macro", zero_division=0)

    importances = {}
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = predict_proba(model, Xp, device=device).argmax(axis=1)
            score = f1_score(y, pred, average="macro", zero_division=0)
            drops.append(base_score - score)
        importances[name] = {"mean_drop": float(np.mean(drops)), "std_drop": float(np.std(drops))}
    return {"baseline_macro_f1": float(base_score), "importances": importances}
