"""Simple, non-MLP baselines the proposed MLP must beat to justify its
existence (task brief §26). All fit on the training partition only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import color_metrics as cm


def nearest_prototype_ciede2000(lab_train, y_train, lab_query, n_classes) -> np.ndarray:
    """Predict the class of the nearest per-class Lab centroid under CIEDE2000."""
    protos = np.stack([lab_train[y_train == k].mean(axis=0) for k in range(n_classes)])
    d = cm.ciede2000(lab_query, protos)  # (N,K)
    return d.argmin(axis=1)


def nearest_centroid_euclidean(X_train, y_train, X_query, n_classes) -> np.ndarray:
    centroids = np.stack([X_train[y_train == k].mean(axis=0) for k in range(n_classes)])
    d = np.linalg.norm(X_query[:, None, :] - centroids[None, :, :], axis=2)
    return d.argmin(axis=1)


def logistic_regression_baseline(X_train, y_train, X_query, class_weight="balanced", seed=42):
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed)
    clf.fit(X_train, y_train)
    return clf.predict(X_query), clf
