"""Statistical utilities. Report effect sizes and intervals, not only p-values; tests are used only where their
assumptions are defensible (paired design, units = independent groups/cartridges, seeds as repeats)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np
from scipy import stats as sps


def mean_sd_ci(x: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    """mean +- sd over seeds and t-based CI for the mean (n small)."""
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else float("nan")
    half = sps.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(m), "sd": float(sd), "ci_lo": float(m - half), "ci_hi": float(m + half), "n": n}


def bootstrap_ci(x: Sequence[float], stat: Callable = np.mean, n_boot: int = 10000, level: float = 0.95, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    q = (1 - level) / 2
    return {"estimate": float(stat(x)), "ci_lo": float(np.quantile(bs, q)), "ci_hi": float(np.quantile(bs, 1 - q))}


def grouped_bootstrap_ci(values: Sequence[float], groups: Sequence, stat: Callable = np.mean, n_boot: int = 5000, level: float = 0.95, seed: int = 0):
    """Cluster bootstrap: resample *groups* (cartridges), not images, so correlated views do not inflate confidence."""
    rng = np.random.default_rng(seed)
    values, groups = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(groups)
    by = {g: values[groups == g] for g in ug}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        bs.append(stat(np.concatenate([by[g] for g in pick])))
    q = (1 - level) / 2
    return {"estimate": float(stat(values)), "ci_lo": float(np.quantile(bs, q)), "ci_hi": float(np.quantile(bs, 1 - q)), "n_groups": len(ug)}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, level: float = 0.95, seed: int = 0) -> Dict[str, float]:
    """Bootstrap CI of the mean paired difference a-b plus a two-sided bootstrap p (fraction of resampled means on the
    far side of 0, doubled). Resample at the level of independent units (see grouped_bootstrap_ci)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    q = (1 - level) / 2
    return {"estimate": float(d.mean()), "ci_lo": float(np.quantile(means, q)), "ci_hi": float(np.quantile(means, 1 - q)),
            "p_two_sided_boot": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison: Wilcoxon signed-rank (+ paired t for reference), Cohen's dz, mean difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"mean_diff": float(d.mean()), "cohens_dz": cohens_dz(a, b), "n": len(d)}
    if len(d) >= 6 and np.any(d != 0):
        out["wilcoxon_p"] = float(sps.wilcoxon(a, b).pvalue)
    else:
        out["wilcoxon_p"] = float("nan")   # exact test cannot reach p<0.05 with n<6 pairs; do not pretend otherwise
    out["paired_t_p"] = float(sps.ttest_rel(a, b).pvalue) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    return out


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    m = int(np.sum(~np.isnan(p)))
    adj = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return {"p_adj": adj, "reject": adj < alpha}


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05) -> Dict[str, np.ndarray]:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return {"p_adj": adj, "reject": adj < q}
