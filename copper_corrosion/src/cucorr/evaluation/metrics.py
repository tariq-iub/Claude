"""Segmentation metrics. Pixels where the *reference* is UNCERTAIN are excluded from every
metric. Predicted UNCERTAIN pixels (abstentions) on valid reference pixels count as errors
for accuracy-type metrics and are reported separately as ``abstention_rate``; selective
metrics (coverage/risk) evaluate only the covered pixels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

from ..schema import CORROSION_IDS, HEALTHY, NUM_CLASSES, UNCERTAIN

K = NUM_CLASSES


def confusion(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(K, K+1) confusion: rows = reference class, cols = predicted class, last col = abstained."""
    v = ref != UNCERTAIN
    r, p = ref[v].astype(int), pred[v].astype(int)
    p = np.where(p == UNCERTAIN, K, p)
    return np.bincount(r * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)


def scores_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm[:, :K]).astype(float)
    fn = cm.sum(1) - tp
    fp = cm[:, :K].sum(0) - tp
    present = cm.sum(1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    iou[~present & (fp == 0)] = np.nan
    dice[~present & (fp == 0)] = np.nan
    freq = cm.sum(1) / max(cm.sum(), 1)
    corr = [c for c in CORROSION_IDS if present[c]]
    return {
        "per_class_iou": iou.tolist(), "per_class_dice": dice.tolist(),
        "per_class_precision": prec.tolist(), "per_class_recall": rec.tolist(),
        "macro_miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
        "macro_dice": float(np.nanmean(dice)) if np.isfinite(dice).any() else float("nan"),
        "fw_dice": float(np.nansum(freq * np.nan_to_num(dice))),
        "corrosion_miou": float(np.nanmean(iou[corr])) if corr else float("nan"),
        "healthy_iou": float(iou[HEALTHY]),
        "balanced_accuracy": float(np.nanmean(rec[present])) if present.any() else float("nan"),
        "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "abstention_rate": float(cm[:, K].sum() / max(cm.sum(), 1)),
    }


def to_binary(mask: np.ndarray) -> np.ndarray:
    out = (mask != HEALTHY).astype(np.uint8)
    out[mask == UNCERTAIN] = UNCERTAIN
    return out


def binary_scores(ref: np.ndarray, pred: np.ndarray) -> dict:
    rb, pb = to_binary(ref), to_binary(pred)
    v = rb != UNCERTAIN
    r, p = rb[v] == 1, pb[v] == 1
    pu = pb[v] == UNCERTAIN
    tp = (r & p).sum(); fp = (~r & p).sum(); fn = (r & ~p & ~pu).sum() + (r & pu).sum()
    tn = (~r & ~p & ~pu).sum()
    d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"binary_dice": d(2 * tp, 2 * tp + fp + fn), "binary_iou": d(tp, tp + fp + fn),
            "sensitivity": d(tp, r.sum()), "specificity": d(tn, (~r).sum()),
            "fpr": d(fp, (~r).sum()), "fnr": d(fn, r.sum())}


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32)
    b = np.zeros(m.shape, bool)
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    b[1:, :] |= m[1:, :] != m[:-1, :]
    return b


def boundary_scores(ref: np.ndarray, pred: np.ndarray, tol: int = 2) -> dict:
    """Boundary F1 at tolerance ``tol`` px and 95th-percentile symmetric boundary distance (HD95)."""
    v = ref != UNCERTAIN
    br, bp = _boundary(ref) & v, _boundary(pred) & v
    if not br.any() and not bp.any():
        return {"boundary_f1": 1.0, "hd95": 0.0}
    if not br.any() or not bp.any():
        return {"boundary_f1": 0.0, "hd95": float("nan")}
    dr, dp = distance_transform_edt(~br), distance_transform_edt(~bp)
    prec = float((dr[bp] <= tol).mean()); rec = float((dp[br] <= tol).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"boundary_f1": f1, "hd95": float(np.percentile(np.concatenate([dr[bp], dp[br]]), 95))}


def small_region_recall(ref: np.ndarray, pred: np.ndarray, max_px: int = 200) -> dict:
    """Fraction of small reference corrosion components (<= max_px) with >=50% pixels correctly labelled."""
    hit = tot = 0
    for c in CORROSION_IDS:
        lab, n = ndi.label(ref == c)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        for i in np.flatnonzero((sizes <= max_px) & (np.arange(len(sizes)) > 0)):
            reg = lab == i
            tot += 1
            hit += (pred[reg] == c).mean() >= 0.5
    return {"small_region_recall": float(hit / tot) if tot else float("nan"), "n_small_regions": int(tot)}


def calibration(ref: np.ndarray, proba: np.ndarray, n_bins: int = 15, max_px: int = 200_000,
                seed: int = 0) -> dict:
    v = ref != UNCERTAIN
    y = ref[v].astype(int); p = proba[v]
    if len(y) > max_px:
        idx = np.random.default_rng(seed).choice(len(y), max_px, replace=False)
        y, p = y[idx], p[idx]
    oh = np.eye(K)[y]
    brier = float(((p - oh) ** 2).sum(1).mean())
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        s = bins == b
        if s.any():
            ece += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return {"brier": brier, "ece": float(ece)}


def coverage_risk(ref: np.ndarray, proba: np.ndarray, n: int = 20) -> list[tuple[float, float]]:
    """Selective-prediction curve: (coverage, error) when keeping the most confident pixels."""
    v = ref != UNCERTAIN
    y = ref[v]; p = proba[v]
    conf = p.max(1); correct = p.argmax(1) == y
    order = np.argsort(-conf)
    cs = np.cumsum(correct[order])
    out = []
    for q in np.linspace(1 / n, 1, n):
        k = max(1, int(q * len(order)))
        out.append((float(q), float(1 - cs[k - 1] / k)))
    return out


def evaluate_segmentation(ref: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None,
                          tol: int = 2, small_max_px: int = 200,
                          rare_classes: tuple[int, ...] | None = None) -> dict:
    """All per-image metrics for one (reference, prediction) pair."""
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {pred.shape}")
    cm = confusion(ref, pred)
    out = scores_from_confusion(cm)
    out["confusion"] = cm.tolist()
    out.update(binary_scores(ref, pred))
    out.update(boundary_scores(ref, pred, tol))
    out.update(small_region_recall(ref, pred, small_max_px))
    # Rare classes must be fixed before testing (from training-set frequencies); passed by the benchmark.
    present = [c for c in (rare_classes or ()) if (ref == c).any()]
    out["rare_class_iou"] = float(np.nanmean([out["per_class_iou"][c] for c in present])) if present else float("nan")
    if proba is not None:
        out.update(calibration(ref, proba))
    return out
