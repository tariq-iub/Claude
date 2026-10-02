"""Metrics: segmentation, detection, severity, polarization-image quality, polarimetric accuracy (circular-safe), calibration."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from scipy.ndimage import gaussian_filter, laplace, sobel, uniform_filter


# ------------------------------------------------------------------ segmentation
class SegMetrics:
    """Streaming confusion-matrix metrics. Pixels are never split between train/test; aggregate over test *images*."""

    def __init__(self, num_classes: int, names: Optional[Sequence[str]] = None, ignore_index: int = -1):
        self.K, self.names, self.ignore = num_classes, list(names or range(num_classes)), ignore_index
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
        keep = target != self.ignore
        idx = target[keep] * self.K + pred[keep]
        self.cm += np.bincount(idx, minlength=self.K ** 2).reshape(self.K, self.K)

    def per_class(self) -> Dict[str, np.ndarray]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        tn = cm.sum() - tp - fp - fn
        eps = 1e-12
        out = {
            "iou": tp / (tp + fp + fn + eps), "dice": 2 * tp / (2 * tp + fp + fn + eps), "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps), "specificity": tn / (tn + fp + eps),
        }
        out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"] + eps)
        out["balanced_accuracy"] = 0.5 * (out["recall"] + out["specificity"])
        out["support"] = cm.sum(1)
        return out

    def summary(self) -> Dict[str, float]:
        pc = self.per_class()
        present = pc["support"] > 0                   # classes absent from the test set are excluded from means
        s = {f"m{k}": float(pc[k][present].mean()) for k in ("iou", "dice", "precision", "recall", "specificity", "f1", "balanced_accuracy")}
        s["pixel_accuracy"] = float(np.diag(self.cm).sum() / max(self.cm.sum(), 1))
        s["n_classes_present"] = int(present.sum())
        return s


def binary_mask_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    m = SegMetrics(2)
    m.update(pred.astype(int), target.astype(int))
    pc = m.per_class()
    return {k: float(pc[k][1]) for k in ("iou", "dice", "precision", "recall", "specificity", "f1", "balanced_accuracy")}


# ------------------------------------------------------------------ detection
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4), b (M,4) as x1,y1,x2,y2."""
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def average_precision(preds: List[Tuple[int, float, np.ndarray]], gts: Dict[int, np.ndarray], iou_thr: float = 0.5) -> float:
    """All-point-interpolated AP for one class. preds: (image_id, score, box); gts: image_id -> (G,4)."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan")
    preds = sorted(preds, key=lambda p: -p[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(preds))
    for i, (img, _, box) in enumerate(preds):
        g = gts.get(img)
        if g is None or len(g) == 0:
            continue
        ious = box_iou(box[None], g)[0]
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not used[img][j]:
            tp[i], used[img][j] = 1, True
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def map_scores(preds_by_class, gts_by_class) -> Dict[str, float]:
    """mAP50 and mAP50-95 (IoU 0.50:0.05:0.95) averaged over classes that have ground truth."""
    thr = np.arange(0.5, 0.96, 0.05)
    aps = np.array([[average_precision(preds_by_class.get(c, []), gts_by_class[c], t) for t in thr] for c in gts_by_class])
    return {"mAP50": float(np.nanmean(aps[:, 0])), "mAP50-95": float(np.nanmean(aps))}


def boxes_from_mask(mask: np.ndarray, min_area: int = 4) -> np.ndarray:
    from scipy.ndimage import label, find_objects
    lab, n = label(mask)
    boxes = []
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is not None and (lab[sl] == i).sum() >= min_area:
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    return np.array(boxes, dtype=float).reshape(-1, 4)


# ------------------------------------------------------------------ severity
def severity_regression(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res, ss_tot = (err ** 2).sum(), ((y_true - y_true.mean()) ** 2).sum()
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def quadratic_weighted_kappa(a, b, n_levels: int) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((n_levels, n_levels))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = (np.arange(n_levels)[:, None] - np.arange(n_levels)[None, :]) ** 2 / (n_levels - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    return float(1 - (W * O).sum() / max((W * E).sum(), 1e-12))


def severity_ordinal(level_true, level_pred, n_levels: int) -> Dict[str, float]:
    t, p = np.asarray(level_true, int), np.asarray(level_pred, int)
    rho = sps.spearmanr(t, p)[0] if len(set(t)) > 1 and len(set(p)) > 1 else float("nan")
    return {"accuracy": float((t == p).mean()), "off_by_one_accuracy": float((np.abs(t - p) <= 1).mean()),
            "qwk": quadratic_weighted_kappa(t, p, n_levels), "spearman": float(rho), "ordinal_mae": float(np.abs(t - p).mean())}


# ------------------------------------------------------------------ polarization-image quality (reference vs processed, grayscale [0,1])
def _grad_mag(x): return np.hypot(sobel(x, 0), sobel(x, 1)) / 8.0


def glare_suppression_ratio(ref: np.ndarray, proc: np.ndarray, thr: float = 0.95) -> float:
    """1 - (glare energy after)/(glare energy before); glare energy = sum max(I-thr,0). 1: all glare removed; <0: glare increased."""
    g0, g1 = np.maximum(ref - thr, 0).sum(), np.maximum(proc - thr, 0).sum()
    return float(1 - g1 / g0) if g0 > 0 else float("nan")


def highlight_area_reduction(ref: np.ndarray, proc: np.ndarray, thr: float = 0.95) -> float:
    a0, a1 = (ref >= thr).mean(), (proc >= thr).mean()
    return float(1 - a1 / a0) if a0 > 0 else float("nan")


def edge_preservation(ref: np.ndarray, proc: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Pearson correlation of gradient-magnitude maps over non-saturated pixels (1 = identical edge structure)."""
    v = (ref < 0.98) & (proc < 0.98) if valid is None else valid
    if v.sum() < 10:
        return float("nan")
    return float(np.corrcoef(_grad_mag(ref)[v], _grad_mag(proc)[v])[0, 1])


def texture_preservation(ref: np.ndarray, proc: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """Ratio of Laplacian variance (processed/reference) over non-saturated pixels; ~1 = texture kept, <1 = smoothed away."""
    v = (ref < 0.98) & (proc < 0.98) if valid is None else valid
    lr, lp = laplace(ref)[v], laplace(proc)[v]
    return float(lp.var() / max(lr.var(), 1e-12))


def local_contrast(x: np.ndarray, k: int = 7) -> float:
    m = uniform_filter(x, k)
    return float(np.sqrt(np.maximum(uniform_filter(x * x, k) - m * m, 0)).mean())


def information_entropy(x: np.ndarray, bins: int = 256) -> float:
    h, _ = np.histogram(np.clip(x, 0, 1), bins=bins, range=(0, 1))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def gradient_preservation(ref: np.ndarray, proc: np.ndarray) -> float:
    gr, gp = _grad_mag(ref), _grad_mag(proc)
    return float(gp.mean() / max(gr.mean(), 1e-12))


def signal_to_glare_ratio(x: np.ndarray, thr: float = 0.95) -> float:
    g = x >= thr
    if g.sum() == 0:
        return float("inf")
    return float(x[~g].mean() / x[g].mean())


# ------------------------------------------------------------------ polarimetric accuracy (circular-safe)
def wrapped_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.mod(a - b, math.pi)
    return np.minimum(d, math.pi - d)


def aolp_error(aolp_hat: np.ndarray, aolp_ref: np.ndarray, weight: Optional[np.ndarray] = None) -> float:
    """Mean wrapped AoLP error (radians, range [0, pi/2]); optional weights (e.g. reference DoLP) because AoLP is
    ill-defined where light is unpolarized. NEVER use the linear difference |a-b|."""
    d = wrapped_diff_np(aolp_hat, aolp_ref)
    return float((d * weight).sum() / max(weight.sum(), 1e-12)) if weight is not None else float(d.mean())


def dolp_np(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.hypot(S[..., 1, :, :], S[..., 2, :, :]) / np.maximum(S[..., 0, :, :], eps), 0, 1)


def aolp_np(S: np.ndarray) -> np.ndarray:
    return 0.5 * np.arctan2(S[..., 2, :, :], S[..., 1, :, :])


def polarimetric_agreement(S_hat: np.ndarray, S_ref: np.ndarray) -> Dict[str, float]:
    """S arrays (3[S0,S1,S2],H,W) (luminance or one channel). Returns S0 relative error, DoLP MAE/RMSE, DoLP-weighted wrapped AoLP error,
    and the 'cos(2 dphi)' agreement in [-1,1]."""
    d_h, d_r = dolp_np(S_hat), dolp_np(S_ref)
    a_h, a_r = aolp_np(S_hat), aolp_np(S_ref)
    w = d_r
    return {
        "s0_rel_err": float(np.abs(S_hat[0] - S_ref[0]).mean() / max(np.abs(S_ref[0]).mean(), 1e-12)),
        "dolp_mae": float(np.abs(d_h - d_r).mean()), "dolp_rmse": float(np.sqrt(((d_h - d_r) ** 2).mean())),
        "aolp_wrapped_err_rad": aolp_error(a_h, a_r, w), "aolp_wrapped_err_deg": float(np.degrees(aolp_error(a_h, a_r, w))),
        "aolp_cos2_agreement": float((np.cos(2 * (a_h - a_r)) * w).sum() / max(w.sum(), 1e-12)),
    }


def image_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    return {"rmse": float(np.sqrt(((a - b) ** 2).mean())), "psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            "ssim": float(structural_similarity(a, b, data_range=1.0))}


# ------------------------------------------------------------------ calibration
def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    conf, correct = np.asarray(conf).ravel(), np.asarray(correct).ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_diagram_data(conf, correct, n_bins: int = 15) -> Dict[str, np.ndarray]:
    conf, correct = np.asarray(conf).ravel(), np.asarray(correct).ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    acc = np.bincount(idx, weights=correct, minlength=n_bins) / np.maximum(cnt, 1)
    mc = np.bincount(idx, weights=conf, minlength=n_bins) / np.maximum(cnt, 1)
    return {"bin_center": 0.5 * (edges[:-1] + edges[1:]), "accuracy": acc, "confidence": mc, "count": cnt}


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    """Multi-class Brier: mean over pixels of sum_k (p_k - 1[y=k])^2. prob (N,K) or (K,...)"""
    p = np.asarray(prob)
    if p.ndim > 2 or (p.ndim == 2 and p.shape[0] != len(np.ravel(target))):
        p = np.moveaxis(p, 1 if p.ndim == 4 else 0, -1).reshape(-1, p.shape[1 if p.ndim == 4 else 0])
    y = np.eye(p.shape[1])[np.ravel(target)]
    return float(((p - y) ** 2).sum(1).mean())


def negative_log_likelihood(prob: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(prob)
    p = np.moveaxis(p, 1 if p.ndim == 4 else 0, -1).reshape(-1, p.shape[1 if p.ndim == 4 else 0]) if p.ndim > 2 else p
    return float(-np.log(np.maximum(p[np.arange(len(p)), np.ravel(target)], eps)).mean())


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray, n_points: int = 50) -> Dict[str, np.ndarray]:
    """Abstain on the most uncertain fraction; risk = error rate among retained. AURC = area under the risk-coverage curve."""
    u, e = np.ravel(uncertainty), np.ravel(error).astype(float)
    order = np.argsort(u)
    e_sorted = e[order]
    cov = np.linspace(1.0 / n_points, 1.0, n_points)
    risk = np.array([e_sorted[: max(1, int(round(c * len(e))))].mean() for c in cov])
    return {"coverage": cov, "risk": risk, "aurc": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(risk, cov))}
