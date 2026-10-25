"""Approach B: supervised pixel classification (RF / SVM / HistGradientBoosting).

Pixels are sampled per image (stratified, capped) for training. Train/test separation is
done at the image (or specimen group) level by the caller - never at the pixel level.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..schema import NUM_CLASSES, UNCERTAIN
from .base import Segmenter
from .features import pixel_features


def sample_pixels(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified sample of flat indices of valid pixels (<= n per image)."""
    flat = mask.ravel()
    out = []
    classes = [c for c in range(NUM_CLASSES) if (flat == c).any()]
    per = max(1, n // max(len(classes), 1))
    for c in classes:
        idx = np.flatnonzero(flat == c)
        out.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    return np.concatenate(out) if out else np.empty(0, int)


def make_classifier(kind: str, cfg_b: dict, seed: int):
    cw = cfg_b.get("class_weight")
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=cfg_b.get("n_estimators", 200), class_weight=cw,
                                      min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True,
                                                   class_weight=cw, random_state=seed))
    if kind == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight=cw,
                                              early_stopping=True, random_state=seed)
    raise ValueError(f"unknown model {kind!r}")


class PixelClassifierSegmenter(Segmenter):
    name = "approach_b_pixel_ml"

    def __init__(self, cfg: dict):
        b = cfg["approach_b"]
        super().__init__(cfg, b.get("abstain_confidence", 0.0))
        self.p = b
        self.name = f"approach_b_{b['model']}"
        self.seed = cfg.get("seed", 0)
        self.clf = None

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> "PixelClassifierSegmenter":
        rng = np.random.default_rng(self.seed)
        n = self.p["pixels_per_image"] if self.p["model"] != "svm" else min(self.p["pixels_per_image"], 800)
        X, y = [], []
        for im, m in zip(images, masks):
            f = pixel_features(self.preprocess(im), self.p["features"])
            f = f.reshape(-1, f.shape[-1])
            idx = sample_pixels(m, n, rng)
            X.append(f[idx]); y.append(m.ravel()[idx])
        X, y = np.concatenate(X), np.concatenate(y)
        if len(np.unique(y)) < 2:
            raise ValueError("need at least two classes in training masks")
        self.clf = make_classifier(self.p["model"], self.p, self.seed).fit(X, y)
        self.classes_ = np.asarray(self.clf.classes_, int)
        return self

    def proba_from_features(self, f: np.ndarray) -> np.ndarray:
        h, w, k = f.shape
        out = np.zeros((h * w, NUM_CLASSES), np.float32)
        flat = f.reshape(-1, k)
        for s in range(0, len(flat), 200_000):
            out[s:s + 200_000][:, self.classes_] = self.clf.predict_proba(flat[s:s + 200_000])
        return out.reshape(h, w, NUM_CLASSES)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit() first")
        return self.proba_from_features(pixel_features(self.preprocess(image), self.p["features"]))

    def feature_importance(self) -> dict | None:
        est = self.clf
        if hasattr(est, "feature_importances_"):
            return {"impurity_importance": est.feature_importances_.tolist()}
        return None
