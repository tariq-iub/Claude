"""Approach C: SLIC superpixels + region classifier + edge-aware graph smoothing."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import slic

from ..schema import NUM_CLASSES, UNCERTAIN
from .base import Segmenter
from .features import pixel_features

REGION_GROUPS = ("rgb", "lab", "hsv", "chroma", "texture")


def superpixels(image: np.ndarray, n_segments: int, compactness: float, seed: int = 0) -> np.ndarray:
    return slic(image, n_segments=n_segments, compactness=compactness, start_label=0,
                convert2lab=True, enforce_connectivity=True)


def region_features(feat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-region mean and std of every pixel feature, plus log region size."""
    n = labels.max() + 1
    idx = np.arange(n)
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n).astype(float)
    out = []
    for k in range(feat.shape[-1]):
        v = feat[..., k].ravel()
        m = np.bincount(flat, v, n) / np.maximum(cnt, 1)
        sq = np.bincount(flat, v * v, n) / np.maximum(cnt, 1)
        out += [m, np.sqrt(np.maximum(sq - m * m, 0))]
    out.append(np.log1p(cnt))
    del idx
    return np.stack(out, 1).astype(np.float32)


def region_adjacency(labels: np.ndarray) -> list[tuple[int, int]]:
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    sel = a != b
    pairs = np.unique(np.sort(np.stack([a[sel], b[sel]], 1), 1), axis=0)
    return [tuple(p) for p in pairs]


def graph_smooth(proba: np.ndarray, mean_lab: np.ndarray, edges, strength: float, iters: int,
                 sigma: float = 10.0) -> np.ndarray:
    """p_i <- (1-s) p_i + s * sum_j w_ij p_j / sum_j w_ij, w_ij = exp(-dE_ij^2 / 2 sigma^2).
    Edge-aware: dissimilar neighbours contribute little, protecting small distinct regions."""
    if strength <= 0 or not edges:
        return proba
    e = np.asarray(edges)
    w = np.exp(-np.sum((mean_lab[e[:, 0]] - mean_lab[e[:, 1]]) ** 2, 1) / (2 * sigma ** 2))
    n = len(proba)
    for _ in range(iters):
        acc = np.zeros_like(proba)
        ws = np.zeros(n)
        np.add.at(acc, e[:, 0], w[:, None] * proba[e[:, 1]])
        np.add.at(acc, e[:, 1], w[:, None] * proba[e[:, 0]])
        np.add.at(ws, e[:, 0], w); np.add.at(ws, e[:, 1], w)
        has = ws > 1e-9
        nb = np.where(has[:, None], acc / np.maximum(ws, 1e-9)[:, None], proba)
        proba = (1 - strength) * proba + strength * nb
    return proba


class SuperpixelSegmenter(Segmenter):
    name = "approach_c_superpixel"

    def __init__(self, cfg: dict):
        c = cfg["approach_c"]
        super().__init__(cfg, c.get("abstain_confidence", 0.0))
        self.p = c
        self.seed = cfg.get("seed", 0)
        self.clf: RandomForestClassifier | None = None

    def _regions(self, image: np.ndarray):
        img = self.preprocess(image)
        labels = superpixels(img, self.p["n_segments"], self.p["compactness"], self.seed)
        feat = pixel_features(img, REGION_GROUPS)
        return labels, region_features(feat, labels), feat

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> "SuperpixelSegmenter":
        X, y = [], []
        for im, m in zip(images, masks):
            labels, rf, _ = self._regions(im)
            n = labels.max() + 1
            for r in range(n):
                vals = m[labels == r]
                vals = vals[vals != UNCERTAIN]
                if len(vals) == 0:
                    continue
                cnt = np.bincount(vals, minlength=NUM_CLASSES)
                if cnt.max() / cnt.sum() >= 0.6:  # skip mixed regions as training targets
                    X.append(rf[r]); y.append(cnt.argmax())
        self.clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=1,
                                          n_jobs=-1, random_state=self.seed).fit(np.array(X), np.array(y))
        self.classes_ = self.clf.classes_.astype(int)
        return self

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit() first")
        labels, rf, feat = self._regions(image)
        pr = np.zeros((len(rf), NUM_CLASSES))
        pr[:, self.classes_] = self.clf.predict_proba(rf)
        # Lab means are features 3..5 in REGION_GROUPS order (rgb 3 then lab 3), stored as mean,std pairs.
        mean_lab = np.stack([rf[:, 2 * k] for k in (3, 4, 5)], 1) * np.array([100, 128, 128])
        pr = graph_smooth(pr, mean_lab, region_adjacency(labels), self.p["smoothing"],
                          self.p["smoothing_iterations"])
        self.last_labels = labels
        return pr[labels].astype(np.float32)
