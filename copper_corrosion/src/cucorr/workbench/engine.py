"""UI-independent engine behind the workbench app.

Splits a benchmark into explicit steps (prepare -> fit -> predict -> evaluate -> compare)
so a UI can show progress and intermediate results. Works both with reference masks
(full evaluation) and without (prediction-only comparison: class composition,
uncertainty, and pairwise agreement between approaches - *not* accuracy).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..config import deep_merge
from ..evaluation.benchmark import ROBUSTNESS, perturb
from ..evaluation.metrics import confusion, coverage_risk, evaluate_segmentation, scores_from_confusion
from ..evaluation.statistics import compare_approaches, mean_ci, select_best
from ..schema import NUM_CLASSES, UNCERTAIN
from ..segmentation.base import Segmenter
from ..segmentation.registry import build

# Display catalogue: key -> (label, base approach, config overrides, needs training, description)
CATALOG: dict[str, dict] = {
    "A · RGB distance": dict(base="approach_a", over={"approach_a": {"color_space": "rgb", "distance": "euclidean"}},
                             desc="Nearest expert reference colour by RGB Euclidean distance (naive baseline)."),
    "A · CIEDE2000": dict(base="approach_a", over={"approach_a": {"color_space": "lab", "distance": "ciede2000"}},
                          desc="Nearest expert reference colour by perceptual CIEDE2000 in CIELAB, with abstention."),
    "B · Random Forest": dict(base="approach_b", over={"approach_b": {"model": "random_forest"}},
                              desc="Pixel classifier on colour + texture features."),
    "B · Gradient Boosting": dict(base="approach_b", over={"approach_b": {"model": "hist_gb"}},
                                  desc="Histogram gradient boosting on the same pixel features."),
    "B · SVM": dict(base="approach_b", over={"approach_b": {"model": "svm"}},
                    desc="RBF support-vector machine (subsampled; slowest of B)."),
    "C · Superpixels": dict(base="approach_c", over={},
                            desc="SLIC superpixels, region classifier, edge-aware graph smoothing."),
    "D · U-Net": dict(base="approach_d", over={}, desc="Deep segmentation (needs PyTorch)."),
    "E · SAM 2 regions": dict(base="approach_e", over={}, desc="Foundation-model regions + colour labels (needs sam2)."),
    "F · Hybrid": dict(base="approach_f", over={}, desc="Ensemble of A, B and C; disagreement drives review."),
}


def available(key: str) -> tuple[bool, str]:
    base = CATALOG[key]["base"]
    if base == "approach_d":
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch not installed (pip install 'cucorr[deep]')"
    if base == "approach_e":
        try:
            import sam2  # noqa: F401
        except ImportError:
            return False, "SAM 2 not installed (pip install 'cucorr[sam]') + checkpoint"
    return True, ""


def make(key: str, cfg: dict) -> Segmenter:
    spec = CATALOG[key]
    return build(spec["base"], deep_merge(cfg, spec["over"]))


@dataclass
class Item:
    image_id: str
    image: np.ndarray
    ref: np.ndarray | None = None
    split: str = "test"


@dataclass
class RunResult:
    key: str
    describe: dict = field(default_factory=dict)
    fit_seconds: float = 0.0
    preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)
    robustness: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def fit(key: str, cfg: dict, train: list[Item]) -> tuple[Segmenter, float]:
    seg = make(key, cfg)
    t0 = time.perf_counter()
    if seg.requires_training:
        if not train:
            raise ValueError("this approach needs training images with reference masks")
        seg.fit([t.image for t in train], [t.ref for t in train])
    return seg, time.perf_counter() - t0


def predict(seg: Segmenter, item: Item):
    t0 = time.perf_counter()
    out = seg.predict_mask(item.image)
    return out, time.perf_counter() - t0


def evaluate(res: RunResult, seg: Segmenter, test: list[Item], cfg: dict, rare: tuple[int, ...]) -> None:
    ev = cfg["evaluation"]
    for it in test:
        if it.ref is None:
            continue
        mask, _, proba = res.preds[it.image_id]
        m = evaluate_segmentation(it.ref, mask, proba if seg.probabilistic else None,
                                  ev["boundary_tolerance_px"], ev["small_region_max_px"], rare)
        m["coverage_risk"] = coverage_risk(it.ref, proba)
        m["confusion_full"] = confusion(it.ref, mask)
        res.metrics[it.image_id] = m


def robustness(res: RunResult, seg: Segmenter, test: list[Item], seed: int = 0) -> None:
    labelled = [t for t in test if t.ref is not None]
    if not labelled:
        return
    rng = np.random.default_rng(seed)
    for kind in ROBUSTNESS:
        cm = sum(confusion(t.ref, seg.predict_mask(perturb(t.image, kind, rng))[0]) for t in labelled)
        res.robustness[kind] = scores_from_confusion(cm)["macro_miou"]


def rare_classes(train: list[Item]) -> tuple[int, ...]:
    masks = [t.ref[t.ref != UNCERTAIN] for t in train if t.ref is not None]
    if not masks:
        return tuple(range(1, NUM_CLASSES))
    frac = np.bincount(np.concatenate(masks), minlength=NUM_CLASSES) / max(sum(m.size for m in masks), 1)
    return tuple(c for c in range(1, NUM_CLASSES) if frac[c] < 0.05) or tuple(range(1, NUM_CLASSES))


SUMMARY = ("macro_miou", "corrosion_miou", "healthy_iou", "binary_dice", "boundary_f1", "rare_class_iou",
           "balanced_accuracy", "sensitivity", "specificity", "abstention_rate", "ece", "small_region_recall")


def summarize(res: RunResult, cfg: dict) -> dict:
    ev = cfg["evaluation"]
    out = {}
    for k in SUMMARY:
        vals = [m.get(k) for m in res.metrics.values() if m.get(k) is not None]
        mu, lo, hi = mean_ci(vals, min(ev["bootstrap_iterations"], 1000), ev["alpha"], cfg.get("seed", 0))
        out[k], out[k + "_lo"], out[k + "_hi"] = mu, lo, hi
    out["seconds_per_image"] = float(np.mean(list(res.seconds.values()))) if res.seconds else float("nan")
    out["fit_seconds"] = res.fit_seconds
    out["review_fraction"] = float(np.mean([(p[0] == UNCERTAIN).mean() for p in res.preds.values()])) \
        if res.preds else float("nan")
    out["robustness_miou"] = float(np.nanmean(list(res.robustness.values()))) if res.robustness else float("nan")
    out["mean_uncertainty"] = float(np.mean([p[1].mean() for p in res.preds.values()])) if res.preds else float("nan")
    return out


def pooled(res: RunResult) -> dict | None:
    if not res.metrics:
        return None
    cm = sum(m["confusion_full"] for m in res.metrics.values())
    return {**scores_from_confusion(cm), "confusion": cm}


def composition(res: RunResult) -> np.ndarray:
    """Mean fraction of pixels per class (+ uncertain as last entry) over predicted images."""
    rows = []
    for mask, _, _ in res.preds.values():
        rows.append([(mask == c).mean() for c in range(NUM_CLASSES)] + [(mask == UNCERTAIN).mean()])
    return np.mean(rows, 0) if rows else np.zeros(NUM_CLASSES + 1)


def pairwise_agreement(results: dict[str, RunResult]) -> tuple[list[str], np.ndarray]:
    """Fraction of pixels on which two approaches predict the same label (uncertain excluded).
    Model-vs-model agreement: says nothing about correctness."""
    keys = [k for k, r in results.items() if r.preds]
    A = np.eye(len(keys))
    for i, j in combinations(range(len(keys)), 2):
        common = set(results[keys[i]].preds) & set(results[keys[j]].preds)
        num = den = 0
        for iid in common:
            a, b = results[keys[i]].preds[iid][0], results[keys[j]].preds[iid][0]
            v = (a != UNCERTAIN) & (b != UNCERTAIN)
            num += (a[v] == b[v]).sum(); den += v.sum()
        A[i, j] = A[j, i] = num / den if den else np.nan
    return keys, A


def statistics(results: dict[str, RunResult], cfg: dict) -> dict | None:
    keyed = {k: r for k, r in results.items() if r.metrics}
    if len(keyed) < 2:
        return None
    ids = sorted(set.intersection(*(set(r.metrics) for r in keyed.values())))
    per = {k: np.array([r.metrics[i]["macro_miou"] for i in ids]) for k, r in keyed.items()}
    return compare_approaches(per, min(cfg["evaluation"]["bootstrap_iterations"], 1000), cfg["evaluation"]["alpha"],
                              cfg.get("seed", 0))


def ranking(results: dict[str, RunResult], cfg: dict) -> list[tuple[str, float]]:
    return select_best({k: summarize(r, cfg) for k, r in results.items() if r.metrics})
