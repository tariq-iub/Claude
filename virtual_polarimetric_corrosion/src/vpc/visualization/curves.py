"""Smoothing/fitting helpers. Rules: (1) measured points are ALWAYS plotted as markers; (2) a fitted line is a separate, labelled
object (never data); (3) raw and fitted values are exported to separate CSVs; (4) the method is recorded in the CSV.
Use a smoother only when justified (many noisy points, a continuous underlying variable); never for categorical x."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter


def loess(x, y, frac: float = 0.5, degree: int = 1, xout=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xout = x if xout is None else np.asarray(xout, float)
    n = len(x)
    k = max(degree + 2, int(np.ceil(frac * n)))
    out = np.empty(len(xout))
    for i, x0 in enumerate(xout):
        d = np.abs(x - x0)
        idx = np.argsort(d)[:k]
        h = max(d[idx].max(), 1e-12)
        w = (1 - (d[idx] / h) ** 3) ** 3
        A = np.vander(x[idx] - x0, degree + 1, increasing=True)
        W = np.sqrt(w)[:, None]
        out[i] = np.linalg.lstsq(A * W, y[idx] * W[:, 0], rcond=None)[0][0]
    return out


def fit_curve(x, y, method: str = "pchip", n_out: int = 200, periodic_period: Optional[float] = None, **kw):
    """Return (x_fine, y_fit). method in {'pchip','spline','savgol','loess','poly','periodic_spline'}."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    xf = np.linspace(x.min(), x.max(), n_out)
    if method == "pchip":
        return xf, PchipInterpolator(x, y)(xf)                      # monotone-preserving, no overshoot
    if method == "spline":
        return xf, UnivariateSpline(x, y, s=kw.get("s", len(x) * np.var(y) * 0.01))(xf)
    if method == "savgol":
        w = kw.get("window", min(len(x) // 2 * 2 - 1, 7)); w = max(w, kw.get("order", 2) + 2 | 1)
        ys = savgol_filter(y, w, kw.get("order", 2))
        return x, ys
    if method == "loess":
        return xf, loess(x, y, kw.get("frac", 0.5), kw.get("degree", 1), xf)
    if method == "poly":
        return xf, np.polyval(np.polyfit(x, y, kw.get("deg", 2)), xf)
    if method == "periodic_spline":
        P = periodic_period or (x.max() - x.min())
        xx = np.append(x, x[0] + P); yy = np.append(y, y[0])
        xf = np.linspace(x[0], x[0] + P, n_out, endpoint=False)
        return xf, CubicSpline(xx, yy, bc_type="periodic")(xf)
    raise ValueError(method)


def export_raw_and_fitted(name: str, x, y, method: str, outdir: str = "tables", series: str = "", **kw):
    """Write <name>_raw.csv (measured points) and <name>_fitted.csv (fit + method). Returns (xf, yf)."""
    os.makedirs(outdir, exist_ok=True)
    xf, yf = fit_curve(x, y, method, **kw)
    pd.DataFrame({"x": x, "y": y, "series": series}).to_csv(os.path.join(outdir, f"{name}_raw.csv"), index=False, mode="a",
                                                            header=not os.path.exists(os.path.join(outdir, f"{name}_raw.csv")))
    pd.DataFrame({"x": xf, "y_fit": yf, "method": method, "series": series}).to_csv(
        os.path.join(outdir, f"{name}_fitted.csv"), index=False, mode="a", header=not os.path.exists(os.path.join(outdir, f"{name}_fitted.csv")))
    return xf, yf
