"""Figure 1 - VP-CorrosionNet framework block diagram (matplotlib patches only; no raster art, no effects)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from .style import INK, INK2, PALETTE, apply_style, save_all

FILL = {"physics": "#eaf2fc", "learned": "#fdeee7", "io": "#f4f4f2", "train": "#efedf8"}
EDGE = {"physics": PALETTE["blue"], "learned": PALETTE["orange"], "io": INK2, "train": PALETTE["violet"]}


def _box(ax, cx, cy, w, h, title, sub, kind, dashed=False):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.008", fc=FILL[kind], ec=EDGE[kind],
                       lw=1.1, ls="--" if dashed else "-", zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy + (0.022 if sub else 0), title, ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(cx, cy - 0.020, sub, ha="center", va="center", fontsize=5.5, color=INK2, zorder=3, linespacing=1.3)
    return (cx, cy, w, h)


def _arrow(ax, a, b, kind="h", color=INK2, ls="-", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=ls, shrinkA=0, shrinkB=0,
                                                    connectionstyle=f"arc3,rad={rad}"), zorder=1)


def _right(b): return (b[0] + b[2] / 2, b[1])
def _left(b): return (b[0] - b[2] / 2, b[1])
def _top(b): return (b[0], b[1] + b[3] / 2)
def _bot(b): return (b[0], b[1] - b[3] / 2)


def build():
    apply_style()
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    W, H = 0.182, 0.105
    xs5 = [0.101, 0.3005, 0.50, 0.6995, 0.899]
    y1, y2, y3 = 0.895, 0.715, 0.535

    ax.text(0.0, 0.985, "Deployment path (conventional RGB camera only)", fontsize=8, fontweight="bold", color=INK, va="top")
    r1 = [
        _box(ax, xs5[0], y1, W, H, "Conventional RGB image", "single frame, sRGB\nno polarization hardware", "io"),
        _box(ax, xs5[1], y1, W, H, "Radiometric linearization", "sRGB → linear radiance\nclipping / glare masks", "physics"),
        _box(ax, xs5[2], y1, W, H, "Geometry / cartridge prior", "silhouette → axis, radius\ncylinder normals n(x)", "physics"),
        _box(ax, xs5[3], y1, W, H, "Diffuse–specular separation", "metal-tinted specular colour;\nboth parts partially polarized", "physics"),
        _box(ax, xs5[4], y1, W, H, "Latent optical state Z", "D, S, n, r, η, k, ρ, φ, g, u\nbounded proxies / distributions", "learned"),
    ]
    for a_, b_ in zip(r1[:-1], r1[1:]):
        _arrow(ax, _right(a_), _left(b_))

    r2 = [
        _box(ax, xs5[0], y2, W, H, "Virtual polarimetric camera", "conductor Fresnel (R_s, R_p, δ),\nGGX, diffuse DoLP → Ŝ0, Ŝ1, Ŝ2", "physics"),
        _box(ax, xs5[1], y2, W, H, "Virtual analyzer θ", "Mueller / Malus, differentiable\nθ ∈ [0°, 180°), period 180°", "physics"),
        _box(ax, xs5[2], y2, W, H, "Virtual polarization stack", "I_θ at 0° … 157.5° (or dense)\nCPE: K plausible Z → mean, var", "physics"),
        _box(ax, xs5[3], y2, W, H, "PSRF", "R(θ) = a0 + a1 cos2θ + b1 sin2θ\n(+ 4θ terms only if justified)", "learned"),
        _box(ax, xs5[4], y2, W, H, "Polarization features", "DoLP_hat, cos/sin 2AoLP_hat,\na0, a1, b1, glare g, uncertainty u", "learned"),
    ]
    zb, cb = r1[-1], r2[0]
    ym = (_bot(zb)[1] + _top(cb)[1]) / 2
    ax.plot([zb[0], zb[0]], [_bot(zb)[1], ym], color=INK2, lw=1.0, zorder=1)
    ax.plot([zb[0], cb[0]], [ym, ym], color=INK2, lw=1.0, zorder=1)
    _arrow(ax, (cb[0], ym), _top(cb))
    for a_, b_ in zip(r2[:-1], r2[1:]):
        _arrow(ax, _right(a_), _left(b_))

    xs4 = [0.1325, 0.3685, 0.6045, 0.8575]
    W3 = 0.2155
    r3 = [
        _box(ax, xs4[0], y3, W3, H, "Multimodal feature fusion", "image RGB + CIELAB / HSV + texture\n+ polarization features (early | mid | late)", "learned"),
        _box(ax, xs4[1], y3, W3, H, "VP-CorrosionNet", "depthwise-separable encoder, ECA,\nedge-aware multi-scale decoder", "learned"),
        _box(ax, xs4[2], y3, W3, H, "Task heads", "segmentation · PittingHead · severity\n(ordinal + area %) · log-variance", "learned"),
        _box(ax, xs4[3], y3, 0.2495, H, "Outputs", "corrosion mask · pitting · severity\nuncertainty / confidence · abstention", "io"),
    ]
    fb, fu = r2[-1], r3[0]
    ym = (_bot(fb)[1] + _top(fu)[1]) / 2
    ax.plot([fb[0], fb[0]], [_bot(fb)[1], ym], color=INK2, lw=1.0, zorder=1)
    ax.plot([fb[0], fu[0]], [ym, ym], color=INK2, lw=1.0, zorder=1)
    _arrow(ax, (fu[0], ym), _top(fu))
    for a_, b_ in zip(r3[:-1], r3[1:]):
        _arrow(ax, _right(a_), _left(b_))

    # training-only lane
    ax.add_patch(FancyBboxPatch((-0.012, 0.045), 1.024, 0.275, boxstyle="round,pad=0,rounding_size=0.01", fc="none", ec=PALETTE["violet"], lw=1.0, ls=(0, (4, 3)), zorder=0))
    ax.text(-0.004, 0.307, "Training only (optional hardware teacher) — not required at deployment", fontsize=7.6, fontweight="bold", color=PALETTE["violet"], va="top")
    yt = 0.165
    t = [
        _box(ax, 0.101, yt, W, H, "Physical polarization camera", "rotating polarizer or\npolarization sensor", "train", True),
        _box(ax, 0.3005, yt, W, H, "0° / 45° / 90° / 135°", "(+ 22.5° … 157.5° subset)\nlinear, registered", "train", True),
        _box(ax, 0.50, yt, W, H, "Teacher representation", "measured S0, S1, S2, DoLP, AoLP\n+ teacher corrosion features", "train", True),
        _box(ax, 0.6995, yt, W, H, "Knowledge distillation", "L_distill: Ŝ vs S, features, logits\ncircular-safe AoLP loss", "train", True),
        _box(ax, 0.899, yt, W, H, "RGB-only student", "same network as the\ndeployed VP-CorrosionNet", "train", True),
    ]
    for a_, b_ in zip(t[:-1], t[1:]):
        _arrow(ax, _right(a_), _left(b_), color=PALETTE["violet"])
    vp = r3[1]
    yc = 0.385
    ax.plot([t[-1][0], t[-1][0]], [_top(t[-1])[1], yc], color=PALETTE["violet"], lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.plot([t[-1][0], vp[0]], [yc, yc], color=PALETTE["violet"], lw=1.0, ls=(0, (3, 2)), zorder=1)
    _arrow(ax, (vp[0], yc), _bot(vp), color=PALETTE["violet"], ls=(0, (3, 2)))
    ax.text(0.62, yc + 0.012, "weights shared: distilled student is what gets deployed", fontsize=5.8, color=PALETTE["violet"], ha="left", va="bottom")

    ax.legend(handles=[Patch(fc=FILL["physics"], ec=EDGE["physics"], label="physics-based (established model; approximations flagged in text)"),
                       Patch(fc=FILL["learned"], ec=EDGE["learned"], label="learned / estimated (proxy latent variables)"),
                       Patch(fc=FILL["io"], ec=EDGE["io"], label="input / output"),
                       Patch(fc=FILL["train"], ec=EDGE["train"], ls="--", label="training-only branch")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=6.4)
    ax.text(0.5, 0.02, "Hatted quantities (Ŝ, DoLP_hat, AoLP_hat) are estimates inferred from RGB under stated assumptions — not measurements.",
            ha="center", fontsize=6.2, color=INK2, style="italic")
    return fig


def make(outdir: str = "figures"):
    return save_all(build(), "fig01_architecture", outdir)


if __name__ == "__main__":
    print(make())
