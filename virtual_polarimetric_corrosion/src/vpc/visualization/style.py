"""Publication style, validated categorical palette, and multi-format export (SVG + PDF + PNG >= 300 dpi)."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-4 of the reference palette (validated with scripts/validate_palette.py: adjacent CVD dE 9.2, normal-vision dE 27.6;
# aqua is < 3:1 contrast on white -> every series also has a distinct marker and a direct label).
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#4a3aa7"}
SERIES = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"]]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI, "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def save_all(fig, name: str, outdir: str = "figures", formats: Iterable[str] = ("svg", "pdf", "png")) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
        paths.append(p)
    plt.close(fig)
    return paths


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def provenance_note(fig, text: str) -> None:
    """Small footer stating data provenance (e.g. SYNTHETIC, N seeds)."""
    fig.text(0.01, -0.01, text, fontsize=6, color=INK2, ha="left", va="top")
