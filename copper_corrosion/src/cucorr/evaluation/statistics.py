"""Paired comparison of approaches on identical held-out images."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Image-level paired bootstrap of mean(a - b). Returns (mean_diff, lo, hi, p_two_sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    centered = boots - d.mean()
    p = float(np.mean(np.abs(centered) >= abs(d.mean())))
    return float(d.mean()), float(lo), float(hi), max(p, 1 / iters)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def compare_approaches(per_image: dict[str, np.ndarray], iters: int = 2000, alpha: float = 0.05,
                       seed: int = 0, min_n_for_p: int = 10) -> dict:
    """per_image: {approach: per-image values of the primary metric, same image order}.
    With fewer than ``min_n_for_p`` paired images, p-values are withheld (NaN): the bootstrap
    distribution of a mean over a handful of images is too coarse to support a test."""
    names = sorted(per_image)
    res, pv = {}, {}
    for a, b in combinations(names, 2):
        x, y = np.asarray(per_image[a], float), np.asarray(per_image[b], float)
        md, lo, hi, p = paired_bootstrap(x, y, iters, alpha, seed)
        ok = np.isfinite(x) & np.isfinite(y)
        try:
            w = stats.wilcoxon(x[ok], y[ok]).pvalue if ok.sum() >= 6 and np.any(x[ok] != y[ok]) else float("nan")
        except ValueError:
            w = float("nan")
        diff = x[ok] - y[ok]
        dz = float(diff.mean() / diff.std(ddof=1)) if ok.sum() > 2 and diff.std(ddof=1) > 0 else float("nan")
        if ok.sum() < min_n_for_p:
            p = float("nan")
        key = f"{a} vs {b}"
        res[key] = {"mean_diff": md, "ci": [lo, hi], "p_bootstrap": p, "p_wilcoxon": float(w),
                    "effect_size_dz": dz, "n_images": int(ok.sum())}
        if np.isfinite(p):
            pv[key] = p
    for k in res:
        res[k]["p_holm"] = float("nan")
    for k, v in holm(pv).items():
        res[k]["p_holm"] = v
    n = min((r["n_images"] for r in res.values()), default=0)
    note = None if n >= 30 else (f"Only {n} paired test images: significance tests are under-powered; "
                                 "report effect sizes and CIs, not 'no difference'.")
    return {"pairwise": res, "power_note": note}


def mean_ci(x, iters: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(iters)]
    return float(x.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def select_best(summary: dict[str, dict], weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Predeclared weighted-rank rule. ``summary[approach][metric]`` = mean value.
    Metrics where lower is better must be prefixed with '-' in ``weights`` keys."""
    weights = weights or {"macro_miou": 0.30, "binary_dice": 0.15, "boundary_f1": 0.15,
                          "rare_class_iou": 0.15, "-ece": 0.05, "-review_fraction": 0.10,
                          "-seconds_per_image": 0.05, "robustness_miou": 0.05}
    names = list(summary)
    score = {n: 0.0 for n in names}
    for key, w in weights.items():
        neg = key.startswith("-"); m = key.lstrip("-")
        vals = {n: summary[n].get(m, np.nan) for n in names}
        finite = [n for n in names if np.isfinite(vals[n])]
        if len(finite) < 2:
            continue
        order = sorted(finite, key=lambda n: vals[n], reverse=not neg)
        for r, n in enumerate(order):
            score[n] += w * (1 - r / (len(order) - 1))
    return sorted(score.items(), key=lambda t: -t[1])
