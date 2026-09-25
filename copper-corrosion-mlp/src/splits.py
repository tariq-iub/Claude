"""Reproducible, stratified data splitting.

labels.xlsx carries no image_id / specimen_id / acquisition_id column and
no exact or near-duplicate rows were found (see data-quality report), so
there is no available grouping structure to exploit for group-disjoint
splitting. This is a documented limitation (task brief §4): row-level
stratified splitting is used, and results should be read as pixel/
observation-level generalization, not necessarily specimen-level
generalization to a *new* piece of copper.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold


def repeated_stratified_folds(y: np.ndarray, n_splits: int = 5, n_repeats: int = 3, seed: int = 42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def train_val_test_split(y: np.ndarray, seed: int = 42, train_frac=0.7, val_frac=0.15):
    """Stratified 70/15/15 split with deterministic largest-remainder-style
    allocation per class."""
    rng = np.random.default_rng(seed)
    n_classes = y.max() + 1
    train_idx, val_idx, test_idx = [], [], []
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)
        n = len(idx_k)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_idx += idx_k[:n_train].tolist()
        val_idx += idx_k[n_train:n_train + n_val].tolist()
        test_idx += idx_k[n_train + n_val:].tolist()
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)
