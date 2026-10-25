"""Plotly figures for the workbench. Colour rules: approaches keep a fixed categorical
slot (by catalogue order, never by rank); magnitudes use one sequential hue; corrosion
classes use the project class palette so charts match the masks."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..schema import CLASS_NAMES, PALETTE, UNCERTAIN
from .engine import CATALOG

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
SEQ = [[0, "#f1f6fd"], [0.5, "#7fb0ea"], [1, "#0d3f80"]]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
FONT = dict(family="Inter, system-ui, sans-serif", size=13, color=INK)


def color_of(key: str) -> str:
    return SERIES[list(CATALOG).index(key) % len(SERIES)] if key in CATALOG else SERIES[-1]


def rgb_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(c)


CLASS_COLORS = [rgb_hex(PALETTE[i]) for i in range(len(CLASS_NAMES))] + [rgb_hex(PALETTE[UNCERTAIN])]


def _layout(fig: go.Figure, title: str = "", height: int = 380, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left",
                                 yref="container", y=0.985, yanchor="top"),
                      font=FONT, height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=20, t=(50 if title else 20) + (34 if kw.get("showlegend", True) else 0), b=10),
                      hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK2))
    return fig


def metric_bars(summ: dict[str, dict], metric: str, label: str) -> go.Figure:
    keys = sorted(summ, key=lambda k: np.nan_to_num(summ[k][metric], nan=-1))
    v = [summ[k][metric] for k in keys]
    lo = [summ[k][metric] - summ[k][metric + "_lo"] for k in keys]
    hi = [summ[k][metric + "_hi"] - summ[k][metric] for k in keys]
    fig = go.Figure(go.Bar(
        y=keys, x=v, orientation="h", marker=dict(color=[color_of(k) for k in keys], line=dict(color=SURFACE, width=2),
                                                  cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=INK2, thickness=1.5, width=4),
        hovertemplate="<b>%{y}</b><br>" + label + " %{x:.3f}<extra></extra>"))
    for k, x, h in zip(keys, v, hi):  # value label beyond the CI whisker so they never collide
        fig.add_annotation(x=(x + np.nan_to_num(h)) if np.isfinite(x) else 0, y=k, xanchor="left", xshift=6,
                           showarrow=False, text=f"{x:.3f}" if np.isfinite(x) else "n/a", font=dict(color=INK))
    _layout(fig, f"{label} (mean over test images, 95% bootstrap CI)", 70 + 46 * len(keys), showlegend=False)
    fig.update_xaxes(range=[0, 1.15], title=None)
    return fig


def metric_heatmap(summ: dict[str, dict], metrics: list[tuple[str, str]]) -> go.Figure:
    keys = list(summ)
    z = np.array([[summ[k][m] for m, _ in metrics] for k in keys], float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[l for _, l in metrics], y=keys, colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
        text=np.vectorize(lambda x: "n/a" if not np.isfinite(x) else f"{x:.2f}")(z), texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=10, outlinewidth=0)))
    _layout(fig, "Metric profile (higher is better)", 80 + 42 * len(keys), showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def per_class_heatmap(per_class: dict[str, list[float]], title="Per-class IoU (pooled test pixels)") -> go.Figure:
    keys = list(per_class)
    z = np.array([per_class[k] for k in keys], float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(CLASS_NAMES), y=keys, colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
        text=np.vectorize(lambda x: "–" if not np.isfinite(x) else f"{x:.2f}")(z), texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=10, outlinewidth=0)))
    _layout(fig, title, 80 + 42 * len(keys), showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def confusion_heatmap(cm: np.ndarray, key: str) -> go.Figure:
    cm = np.asarray(cm, float)
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cols = list(CLASS_NAMES) + ["Abstained"]
    fig = go.Figure(go.Heatmap(
        z=rown, x=cols, y=list(CLASS_NAMES), colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
        customdata=cm, text=np.vectorize(lambda x: f"{x:.0%}" if x >= 0.005 else "")(rown), texttemplate="%{text}",
        hovertemplate="Reference <b>%{y}</b> → predicted <b>%{x}</b><br>%{z:.1%} of row (%{customdata:,.0f} px)"
                      "<extra></extra>", colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%")))
    _layout(fig, f"Confusion matrix · {key} (row-normalised)", 420, showlegend=False)
    fig.update_yaxes(autorange="reversed", title="Reference")
    fig.update_xaxes(title="Predicted")
    return fig


def robustness_heatmap(rob: dict[str, dict[str, float]], clean: dict[str, float]) -> go.Figure:
    """Approach x condition grid; hover shows the change from clean images."""
    keys = [k for k in rob if rob[k]]
    conds = ["clean"] + list(next(iter(rob[k] for k in keys)))
    z = np.array([[clean.get(k, np.nan)] + list(rob[k].values()) for k in keys], float)
    delta = z - z[:, :1]
    fig = go.Figure(go.Heatmap(
        z=z, x=conds, y=keys, colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2, customdata=delta,
        text=np.vectorize(lambda x: f"{x:.2f}")(z), texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> · %{x}<br>mIoU %{z:.3f} (Δ %{customdata:+.3f} vs clean)<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0)))
    _layout(fig, "Robustness to acquisition changes (macro mIoU)", 100 + 42 * len(keys), showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def coverage_risk_lines(curves: dict[str, list]) -> go.Figure:
    fig = go.Figure()
    for k, c in curves.items():
        c = np.asarray(c)
        fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], name=k, mode="lines", line=dict(color=color_of(k), width=2),
                                 hovertemplate="<b>" + k + "</b><br>coverage %{x:.0%}<br>error %{y:.3f}<extra></extra>"))
    _layout(fig, "Selective prediction: error vs coverage (keep most confident pixels)", 400, hovermode="x unified")
    fig.update_xaxes(tickformat=".0%", title="Coverage")
    fig.update_yaxes(title="Pixel error rate", rangemode="tozero")
    return fig


def speed_accuracy(summ: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    pos = ["top center", "bottom center", "middle right", "middle left"]
    order = sorted(summ, key=lambda k: summ[k]["seconds_per_image"])
    for k, s in summ.items():
        fig.add_trace(go.Scatter(x=[s["seconds_per_image"]], y=[s["macro_miou"]], name=k, mode="markers+text",
                                 text=[k], textposition=pos[order.index(k) % 4], cliponaxis=False, textfont=dict(color=INK2, size=11),
                                 marker=dict(size=14, color=color_of(k), line=dict(color=SURFACE, width=2)),
                                 hovertemplate="<b>" + k + "</b><br>%{x:.2f} s/image<br>mIoU %{y:.3f}<extra></extra>"))
    _layout(fig, "Accuracy vs inference cost", 400, showlegend=False)
    fig.update_xaxes(type="log", title="Seconds per image (log)")
    fig.update_yaxes(title="Macro mIoU", range=[0, 1.05])
    return fig


def composition_bars(comp: dict[str, np.ndarray]) -> go.Figure:
    keys = list(comp)
    names = list(CLASS_NAMES) + ["Uncertain"]
    fig = go.Figure()
    for c, n in enumerate(names):
        fig.add_trace(go.Bar(y=keys, x=[comp[k][c] for k in keys], name=n, orientation="h",
                             marker=dict(color=CLASS_COLORS[c], line=dict(color=SURFACE, width=2)),
                             hovertemplate="<b>%{y}</b><br>" + n + ": %{x:.1%}<extra></extra>"))
    _layout(fig, "Predicted class composition (mean pixel fraction)", 90 + 42 * len(keys), barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return fig


def agreement_heatmap(keys: list[str], A: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=A, x=keys, y=keys, colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
        text=np.vectorize(lambda x: "–" if not np.isfinite(x) else f"{x:.0%}")(A), texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}<br>same label on %{z:.1%} of pixels<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickformat=".0%")))
    _layout(fig, "Pairwise agreement between approaches (not accuracy)", 120 + 40 * len(keys), showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def swatch_figure(classes: list[dict]) -> go.Figure:
    fig = go.Figure()
    for r, c in enumerate(classes):
        for k, rgb in enumerate([c["rep_color"], *c["refs"]]):
            fig.add_shape(type="rect", x0=k, x1=k + 0.92, y0=-r, y1=-r + 0.86, fillcolor=rgb_hex(rgb),
                          line=dict(width=0), layer="below")
            fig.add_trace(go.Scatter(x=[k + 0.46], y=[-r + 0.43], mode="markers", marker=dict(size=40, opacity=0),
                                     showlegend=False,
                                     hovertemplate=f"<b>{c['name']}</b> {'rep' if k == 0 else 'ref' + str(k)}"
                                                   f"<br>RGB {tuple(rgb)}<extra></extra>"))
    _layout(fig, "", 60 * len(classes) + 40, showlegend=False)
    fig.update_xaxes(range=[-0.05, 4], tickvals=[0.46, 1.46, 2.46, 3.46], ticktext=["rep_color", "ref1", "ref2", "ref3"],
                     showgrid=False, side="top")
    fig.update_yaxes(range=[-len(classes) + 0.9, 0.9], tickvals=[-r + 0.43 for r in range(len(classes))],
                     ticktext=[c["name"] for c in classes], showgrid=False)
    return fig


def manifest_scatter(m) -> go.Figure:
    fig = go.Figure()
    for i, n in enumerate(CLASS_NAMES):
        s = m[m.annotation_title == n]
        fig.add_trace(go.Scatter(x=s.lab_a, y=s.lab_b, mode="markers", name=n,
                                 marker=dict(size=8, color=CLASS_COLORS[i], line=dict(color=SURFACE, width=1),
                                             opacity=0.85),
                                 customdata=np.stack([s.cluster_id, s.lab_l], 1),
                                 hovertemplate=f"<b>{n}</b><br>cluster %{{customdata[0]}}<br>L* %{{customdata[1]:.1f}}"
                                               "<br>a* %{x:.1f} · b* %{y:.1f}<extra></extra>"))
    _layout(fig, "Manifest cluster centroids in the a*b* plane", 460)
    fig.update_xaxes(title="a*  (green ← → red)")
    fig.update_yaxes(title="b*  (blue ← → yellow)", scaleanchor="x")
    return fig
