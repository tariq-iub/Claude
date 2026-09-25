"""Minimal but real unit tests (pytest) covering the scientifically
critical invariants: CIEDE2000 correctness, train-only scaler leakage
prevention, metric correctness on a known confusion matrix, and
feature-set shapes."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color_metrics import ciede2000, srgb_to_lab, rgb_to_hsv
from src.preprocessing import StandardScalerTrainOnly
from src.evaluator import compute_metrics
from src.features import build_features, FEATURE_SETS
from src.losses import compute_class_weights


def test_ciede2000_sharma_reference_cases():
    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0009), 7.1792),
        ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ]
    for l1, l2, expected in pairs:
        d = ciede2000(np.array(l1), np.array(l2))[0, 0]
        assert abs(d - expected) < 0.01


def test_ciede2000_identity_is_zero():
    lab = np.array([[50.0, 10.0, -20.0]])
    d = ciede2000(lab, lab)
    assert d[0, 0] < 1e-8


def test_scaler_fit_transform_train_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(50, 3))
    X_test = rng.normal(loc=50.0, scale=1.0, size=(10, 3))  # deliberately shifted
    scaler = StandardScalerTrainOnly().fit(X_train)
    z_train = scaler.transform(X_train)
    assert abs(z_train.mean()) < 0.3
    # test data, being off-distribution, must NOT renormalize to mean 0 --
    # proves scaler statistics come from train only.
    z_test = scaler.transform(X_test)
    assert z_test.mean() > 5.0


def test_scaler_constant_feature_handled():
    X = np.ones((10, 2))
    X[:, 1] = np.arange(10)
    scaler = StandardScalerTrainOnly().fit(X)
    z = scaler.transform(X)
    assert np.all(z[:, 0] == 0.0)
    assert scaler.constant_mask_[0]


def test_metrics_known_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y_true, y_pred, 3, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    cm = np.array(m["confusion_matrix"])
    assert cm[0].sum() == 2 and cm[1].sum() == 2 and cm[2].sum() == 2


def test_feature_sets_produce_expected_dims():
    lab = np.array([[50.0, 10.0, -5.0], [30.0, -2.0, 8.0]])
    dims = {"lab": 3, "lab_hsv": 7, "extended": 9}
    for fset, dim in dims.items():
        X, names = build_features(lab, fset)
        assert X.shape == (2, dim)
        assert len(names) == dim


def test_class_weights_mean_one():
    y = np.array([0, 0, 0, 1, 2])
    w = compute_class_weights(y, 3, method="inverse-frequency")
    assert abs(w.mean() - 1.0) < 1e-8


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", __file__, "-v"])
