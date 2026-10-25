"""Inter-annotator agreement (human vs human). Not evidence of chemical validity.

Pixels are highly spatially correlated, so pixel-level kappa CIs must be computed by
resampling *images* (cluster bootstrap), as done in :func:`bootstrap_ci`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from ..schema import NUM_CLASSES, UNCERTAIN
from .metrics import binary_scores, boundary_scores, confusion, scores_from_confusion, to_binary


def cohen_kappa_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def pair_cm(a: Sequence[np.ndarray], b: Sequence[np.ndarray], k: int = NUM_CLASSES) -> np.ndarray:
    """K x K confusion over pixels both annotators labelled (Uncertain excluded)."""
    cm = np.zeros((k, k), np.int64)
    for x, y in zip(a, b):
        v = (x != UNCERTAIN) & (y != UNCERTAIN)
        cm += np.bincount(x[v].astype(int) * k + y[v], minlength=k * k).reshape(k, k)
    return cm


def fleiss_kappa(masks_by_rater: Sequence[Sequence[np.ndarray]], k: int = NUM_CLASSES,
                 max_px_per_image: int = 50_000, seed: int = 0) -> float:
    """Fleiss' kappa over pixels labelled by all raters (pixels subsampled per image)."""
    rng = np.random.default_rng(seed)
    rows = []
    for imgs in zip(*masks_by_rater):
        st = np.stack([m.ravel() for m in imgs])
        v = (st != UNCERTAIN).all(0)
        idx = np.flatnonzero(v)
        if len(idx) > max_px_per_image:
            idx = rng.choice(idx, max_px_per_image, replace=False)
        sub = st[:, idx]
        rows.append(np.stack([(sub == c).sum(0) for c in range(k)], 1))
    if not rows:
        return float("nan")
    M = np.concatenate(rows).astype(float)
    n = M.sum(1)[0]
    if n < 2:
        return float("nan")
    P_i = ((M * (M - 1)).sum(1)) / (n * (n - 1))
    p_j = M.sum(0) / M.sum()
    Pe = (p_j ** 2).sum()
    return float((P_i.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def disagreement_map(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel fraction of raters disagreeing with the plurality label (Uncertain counts as a vote)."""
    st = np.stack(masks).astype(int)
    st = np.where(st == UNCERTAIN, NUM_CLASSES, st)
    counts = np.stack([(st == c).sum(0) for c in range(NUM_CLASSES + 1)])
    return (1 - counts.max(0) / len(masks)).astype(np.float32)


def bootstrap_ci(stat_fn, n_items: int, iters: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of ``stat_fn(indices)`` with image-level resampling."""
    rng = np.random.default_rng(seed)
    vals = [stat_fn(rng.integers(0, n_items, n_items)) for _ in range(iters)]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def calculate_inter_annotator_agreement(annotations: dict[str, Sequence[np.ndarray]],
                                        iters: int = 1000, seed: int = 0, tol: int = 2) -> dict:
    """annotations: {rater_id: [mask_img0, mask_img1, ...]} aligned by image index."""
    raters = sorted(annotations)
    if len(raters) < 2:
        raise ValueError("need at least two annotators")
    n = len(annotations[raters[0]])
    if any(len(annotations[r]) != n for r in raters):
        raise ValueError("all annotators must label the same images")
    out: dict = {"raters": raters, "n_images": n, "pairs": {}}
    for a, b in combinations(raters, 2):
        A, B = annotations[a], annotations[b]
        cm = pair_cm(A, B)
        full = confusion(np.concatenate([x.ravel() for x in A]),
                         np.concatenate([x.ravel() for x in B]))
        sc = scores_from_confusion(full)
        bA = [to_binary(x) for x in A]; bB = [to_binary(x) for x in B]
        bcm = pair_cm(bA, bB, 2)
        bnd = [boundary_scores(x, y, tol)["boundary_f1"] for x, y in zip(A, B)]
        out["pairs"][f"{a}|{b}"] = {
            "cohen_kappa": cohen_kappa_from_cm(cm),
            "cohen_kappa_ci": bootstrap_ci(lambda ix: cohen_kappa_from_cm(pair_cm([A[i] for i in ix],
                                                                                  [B[i] for i in ix])),
                                           n, iters, seed=seed),
            "binary_kappa": cohen_kappa_from_cm(bcm),
            "per_class_iou": sc["per_class_iou"], "per_class_dice": sc["per_class_dice"],
            "per_class_precision_b_wrt_a": sc["per_class_precision"],
            "per_class_recall_b_wrt_a": sc["per_class_recall"],
            "binary": binary_scores(np.concatenate([x.ravel() for x in A]),
                                    np.concatenate([x.ravel() for x in B])),
            "boundary_f1_mean": float(np.mean(bnd)),
            "confusion": cm.tolist(),
            "uncertain_rate": {a: float(np.mean([(x == UNCERTAIN).mean() for x in A])),
                               b: float(np.mean([(x == UNCERTAIN).mean() for x in B]))},
        }
    if len(raters) > 2:
        out["fleiss_kappa"] = fleiss_kappa([annotations[r] for r in raters], seed=seed)
    return out
