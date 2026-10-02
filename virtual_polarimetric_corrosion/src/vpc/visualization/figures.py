"""Publication figures 2-15 (Figure 1 lives in fig01_architecture.py).

Figures 3, 4, 5, 6, 9 and the fallback of 2 are generated from the *physics model and the synthetic renderer* (no real data);
they are labelled accordingly in a provenance footer. Figures 7, 8, 10-15 read RESULT files produced by real experiments
(results/*.csv, results/predictions/*.npz). If those files are absent the function returns None and prints why; nothing is
fabricated. Measured points are always drawn as markers; any fitted line is a separate object and its raw/fitted values are
exported as CSV (see curves.export_raw_and_fitted).
"""
from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

from ..corrosion.segmentation import CLASS_NAMES
from ..experiments import metrics as M
from ..experiments.synthetic import SynthConfig, make_group, render_view
from ..models.baseline import physics_only_virtual_polarizer
from ..optics.fresnel import fresnel_reflectance, preset
from ..optics.reflection import diffuse_dolp, roughness_depolarization
from ..polarization.angle_search import optimise_analyzer
from ..polarization.virtual_analyzer import VirtualAnalyzer
from .curves import export_raw_and_fitted, fit_curve
from .style import GRID, INK, INK2, MARKERS, PALETTE, SERIES, apply_style, panel_label, provenance_note, save_all

SYNTH_NOTE = "SYNTHETIC data / physics model only (illustrative proxy optical constants) - not evidence about real cartridges."
DIVERGING = LinearSegmentedColormap.from_list("div", [PALETTE["blue"], "#f2f2f0", PALETTE["orange"]])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", PALETTE["blue"], "#10305a"])
CLASS_COLORS = ["#8a8a85", "#c9a24b", "#d9ad8c", "#6b3f2a", "#1baf7a", "#e8e4cf", "#2a2a2a"]
CLASS_CMAP = matplotlib.colors.ListedColormap(CLASS_COLORS)
RGB_NAMES = ("R", "G", "B")


def _lum(x):  # (3,H,W) linear -> (H,W)
    return 0.2126729 * x[0] + 0.7151522 * x[1] + 0.0721750 * x[2]


def _synthetic_sample(seed_group=11, seed_view=3, size=128, mode="environment", level=0.7, **kw):
    cfg = SynthConfig(size=size, **kw)
    return render_view(make_group(seed_group, cfg, level=level), seed_view, cfg, force_mode=mode)


def _most_polarized_view(n_try: int = 24, **kw):
    """Demonstration-sample selection RULE (stated in figure footers): among the first n_try views of one synthetic cartridge, take
    the view whose top-1 % pixels carry the largest polarized radiance sqrt(S1^2+S2^2). Used only for illustrating modules."""
    cfg = SynthConfig(size=kw.pop("size", 128), **kw)
    grp = make_group(11, cfg, level=0.7)
    best, best_v = None, -1.0
    for v in range(1, n_try + 1):
        d = render_view(grp, v, cfg, force_mode="environment")
        S = d["S_gt"].numpy()
        pol = np.hypot(S[:, 1], S[:, 2]).mean(0)
        val = float(np.sort(pol[d["obj_mask"].numpy()])[-max(1, int(0.01 * d["obj_mask"].sum())):].mean())
        if val > best_v:
            best, best_v = d, val
    return best


def _need(path: str) -> bool:
    if not os.path.exists(path):
        print(f"SKIPPED: requires {path} (produced by the experiment scripts); no placeholder is generated.")
        return False
    return True


def _csv(results: str, name: str) -> Optional[pd.DataFrame]:
    p = os.path.join(results, f"{name}.csv")
    if not _need(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        print(f"SKIPPED: {p} has no rows.")
        return None
    return df


# --------------------------------------------------------------------------------------- Figure 3
def fig03_polarization_physics(outdir="figures", **_):
    apply_style()
    th = torch.linspace(0, math.pi / 2 - 1e-4, 400, dtype=torch.float64)
    deg = np.degrees(th.numpy())
    n, k = preset("copper", torch.float64)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    a = ax[0, 0]
    Rs, Rp = (lambda r: (r[0], r[1]))(fresnel_reflectance(th, 1.5, 0.0))
    a.plot(deg, Rs, color=SERIES[0], label="$R_s$ dielectric n=1.5"); a.plot(deg, Rp, color=SERIES[0], ls="--", label="$R_p$ dielectric")
    Rs, Rp, _d = fresnel_reflectance(th, n[1], k[1])
    a.plot(deg, Rs, color=SERIES[1], label="$R_s$ copper (G)"); a.plot(deg, Rp, color=SERIES[1], ls="--", label="$R_p$ copper (G)")
    a.set(xlabel="incidence angle (deg)", ylabel="intensity reflectance", title="Fresnel reflectance (model)"); a.legend(frameon=False, loc="upper left")
    panel_label(a, "a")
    a = ax[0, 1]
    for i, (c, nm) in enumerate(zip(SERIES, ("R", "G", "B"))):
        Rs, Rp, _d = fresnel_reflectance(th, n[i], k[i]); a.plot(deg, ((Rs - Rp) / (Rs + Rp)).numpy(), color=c, label=f"copper {nm}")
    Rs, Rp = fresnel_reflectance(th, 1.5, 0.0)[:2]; a.plot(deg, ((Rs - Rp) / (Rs + Rp)).numpy(), color=INK2, ls=":", label="dielectric n=1.5")
    a.set(xlabel="incidence angle (deg)", ylabel="specular DoLP (unpolarized in)", title="Specular DoLP: conductor vs dielectric (model)"); a.legend(frameon=False)
    panel_label(a, "b")
    a = ax[1, 0]
    for i, (c, nm) in enumerate(zip(SERIES, RGB_NAMES)):
        Rs, Rp, d = fresnel_reflectance(th, n[i], k[i]); a.plot(deg, np.degrees(d.numpy()), color=c, label=f"copper {nm}")
    a.set(xlabel="incidence angle (deg)", ylabel="relative retardance δ (deg)", title="Retardance on metal reflection (model)"); a.legend(frameon=False)
    panel_label(a, "c")
    a = ax[1, 1]
    zen = th.float()
    for nn_, c in zip((1.4, 1.8, 2.5), SERIES):
        a.plot(deg, diffuse_dolp(zen, nn_).numpy(), color=c, label=f"diffuse, n={nn_}")
    r = torch.tensor([0.05, 0.3, 0.6])
    a.set(xlabel="zenith angle (deg)", ylabel="diffuse DoLP", title="Diffuse polarization (Atkinson–Hancock, model)"); a.legend(frameon=False)
    panel_label(a, "d")
    fig.tight_layout()
    provenance_note(fig, "Analytic model curves (lines), copper constants are RGB-effective proxies, not measurements of inspected cartridges.")
    return save_all(fig, "fig03_polarization_physics", outdir)


# --------------------------------------------------------------------------------------- Figure 4
def fig04_virtual_analyzer_sweep(outdir="figures", tables="tables", **_):
    apply_style()
    d = _most_polarized_view()
    S = d["S_gt"].unsqueeze(0)
    va = VirtualAnalyzer()
    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    stack = va(S, angles)[0]
    obj, mask = d["obj_mask"].numpy(), d["mask"].numpy()
    ys, xs = np.nonzero(obj)
    sl = (slice(ys.min() - 2, ys.max() + 3), slice(xs.min() - 2, xs.max() + 3))
    vmax = float(np.quantile(stack.numpy()[:, :, obj].ravel(), 0.995))
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.9], hspace=0.28, wspace=0.05)
    for i, a_ in enumerate(angles):
        ax = fig.add_subplot(gs[i // 4, i % 4]); ax.axis("off")
        im = np.clip(stack[i].permute(1, 2, 0).numpy() / max(vmax, 1e-3), 0, 1) ** (1 / 2.2)
        ax.imshow(im[sl]); ax.set_title(f"θ = {a_:g}°", fontsize=7, pad=2)
    ax = fig.add_subplot(gs[2, :])
    dense = np.arange(0, 180, 2.5)
    dstack = va(S, dense)[0]
    S_np = d["S_gt"].numpy()
    amp = np.hypot(S_np[:, 1], S_np[:, 2]).mean(0)                      # polarized radiance per pixel
    picks = {}
    for name, cls in (("max polarized radiance (any class)", None), ("healthy metal (max modulation)", 1), ("patina (max modulation)", 4)):
        m = obj if cls is None else (mask == cls) & (amp > 0)
        if m.any():
            picks[name] = np.unravel_index(np.argmax(np.where(m, amp, -1)), amp.shape)
    for (nm, (y, x)), c, mk in zip(picks.items(), SERIES, MARKERS):
        ax.plot(dense, _lum(dstack[:, :, y, x].T.numpy()), color=c, lw=1.0, alpha=0.9)
        ax.plot(angles, _lum(stack[:, :, y, x].T.numpy()), mk, color=c, ms=5, label=f"{nm}")
    ax.set(xlabel="virtual analyzer angle θ (deg)", ylabel="luminance I_θ (linear)", xlim=(-3, 182),
           title="Analyzer response of three pixels: I_θ = ½(S0 + S1 cos2θ + S2 sin2θ), period 180°")
    ax.legend(frameon=False, ncol=1, loc="center right", fontsize=6.5)
    provenance_note(fig, SYNTH_NOTE + " View chosen by rule (most polarized highlight of 24 views). Markers: 8 canonical angles; line: dense sweep of the same model.")
    return save_all(fig, "fig04_virtual_analyzer_sweep", outdir)


# --------------------------------------------------------------------------------------- Figure 5
def fig05_stokes_dolp_aolp(outdir="figures", **_):
    apply_style()
    d = _most_polarized_view()
    S = d["S_gt"].numpy()
    L = 0.2126729 * S[0] + 0.7151522 * S[1] + 0.0721750 * S[2]            # luminance Stokes (3,H,W)
    s0, s1, s2 = L
    dolp = np.clip(np.hypot(s1, s2) / np.maximum(s0, 1e-6), 0, 1)
    aolp = 0.5 * np.arctan2(s2, s1)
    obj = d["obj_mask"].numpy()
    ys, xs = np.nonzero(obj)
    sl = (slice(ys.min() - 2, ys.max() + 3), slice(xs.min() - 2, xs.max() + 3))
    fig, ax = plt.subplots(2, 3, figsize=(6.6, 4.4))
    items = [("RGB (input)", None, None), ("S0 (luminance, linear)", s0, "gray"), ("DoLP", dolp, SEQ),
             ("S1", s1, DIVERGING), ("S2", s2, DIVERGING), ("AoLP (where DoLP > 0.05)", aolp, "twilight")]
    for a_, (t, im, cm) in zip(ax.ravel(), items):
        a_.axis("off"); a_.set_title(t, fontsize=7.5)
        if im is None:
            a_.imshow(np.where(obj[..., None], d["rgb"].permute(1, 2, 0).numpy(), 0.5)[sl]); continue
        im = np.where(obj, im, np.nan)
        if t in ("S1", "S2"):
            v = np.nanpercentile(np.abs(im), 99); h = a_.imshow(im[sl], cmap=cm, vmin=-v, vmax=v)
        elif t.startswith("AoLP"):
            h = a_.imshow(np.where(dolp > 0.05, im, np.nan)[sl], cmap=cm, vmin=-math.pi / 2, vmax=math.pi / 2)
        elif t == "DoLP":
            h = a_.imshow(im[sl], cmap=cm, vmin=0, vmax=max(0.1, float(np.nanpercentile(im, 99.5))))
        else:
            h = a_.imshow(im[sl], cmap=cm, vmin=0, vmax=float(np.nanpercentile(im, 99.5)))
        plt.colorbar(h, ax=a_, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    fig.tight_layout()
    provenance_note(fig, SYNTH_NOTE + " Renderer ground truth. Copper polarizes weakly in the visible except near grazing angles (cartridge limb); AoLP has period π.")
    return save_all(fig, "fig05_stokes_dolp_aolp", outdir)


# --------------------------------------------------------------------------------------- Figure 2
def fig02_physical_vs_virtual(outdir="figures", results="results", **_):
    """Uses results/predictions/*.npz (key S_hat__<method>) when available; otherwise falls back to the untrained physics-only
    virtual polarizer on a synthetic sample, which shows the gap that the learned/distilled models must close."""
    apply_style()
    d = _most_polarized_view()
    S_true = d["S_gt"].numpy()
    L = lambda S: 0.2126729 * S[0] + 0.7151522 * S[1] + 0.0721750 * S[2]
    r = physics_only_virtual_polarizer(d["rgb"].unsqueeze(0), d["normals_prior"].unsqueeze(0), 45.0)
    S_hat = r["S_hat"][0].numpy()
    label_hat = "physics-only virtual (untrained)"
    note = SYNTH_NOTE
    Lt, Lh = L(S_true), L(S_hat)
    scale = (Lt[0][d["obj_mask"].numpy()].mean() / max(Lh[0][d["obj_mask"].numpy()].mean(), 1e-9))
    Lh = Lh * scale                                     # exposure-normalise the unknown absolute scale
    obj = d["obj_mask"].numpy()
    f = lambda s: (np.clip(np.hypot(s[1], s[2]) / np.maximum(s[0], 1e-6), 0, 1), 0.5 * np.arctan2(s[2], s[1]))
    dt, at = f(Lt); dh, ah = f(Lh)
    err = np.degrees(M.wrapped_diff_np(ah, at))
    w = dt * obj
    fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.9))
    ax = ax.ravel()
    ax[0].imshow(d["rgb"].permute(1, 2, 0).numpy()); ax[0].set_title("input RGB")
    ax[1].imshow(np.where(obj, dt, np.nan), cmap=SEQ, vmin=0, vmax=1); ax[1].set_title("DoLP (renderer truth)")
    ax[2].imshow(np.where(obj, dh, np.nan), cmap=SEQ, vmin=0, vmax=1); ax[2].set_title("DoLP_hat (" + "physics-only" + ")")
    ax[3].imshow(np.where(obj, np.abs(dh - dt), np.nan), cmap="Greys", vmin=0, vmax=0.5); ax[3].set_title("|ΔDoLP|")
    ax[4].imshow(np.where(obj & (dt > 0.05), at, np.nan), cmap="twilight", vmin=-math.pi / 2, vmax=math.pi / 2); ax[4].set_title("AoLP (truth)")
    ax[5].imshow(np.where(obj & (dt > 0.05), ah, np.nan), cmap="twilight", vmin=-math.pi / 2, vmax=math.pi / 2); ax[5].set_title("AoLP_hat")
    ax[6].imshow(np.where(obj & (dt > 0.05), err, np.nan), cmap="Greys", vmin=0, vmax=90); ax[6].set_title("wrapped AoLP error (deg)")
    ax[7].axis("off")
    pa = M.polarimetric_agreement(Lh, Lt)
    ax[7].text(0, 0.9, f"{label_hat}\n\nDoLP MAE = {pa['dolp_mae']:.3f}\nAoLP wrapped err = {pa['aolp_wrapped_err_deg']:.1f}°\n(DoLP-weighted)\ncos2Δφ agreement = {pa['aolp_cos2_agreement']:.2f}",
               va="top", fontsize=7)
    for a in ax[:7]:
        a.axis("off")
    provenance_note(fig, note + " One sample; replace with held-out hardware measurements (results/predictions) for any claim.")
    return save_all(fig, "fig02_physical_vs_virtual", outdir)


# --------------------------------------------------------------------------------------- Figure 6
def fig06_rgb_vs_virtual_crosspol(outdir="figures", **_):
    """RGB vs analyzer-only vs virtual source+analyzer (parallel / crossed) images.  The virtual images use the renderer's TRUE
    latent state, i.e. they show what the Mueller-chain physics predicts when the latent estimate is perfect (an upper bound,
    NOT an independent validation: renderer and chain share the same Fresnel model)."""
    apply_style()
    from ..color import linear_to_srgb
    from ..inverse.latent_optics import PhysicsStokesLayer
    d = _most_polarized_view(exposure_range=(1.3, 1.8))
    lg = d["latent_gt"]
    z = {"n": d["normals"].unsqueeze(0), "D": lg["D"].unsqueeze(0), "S": lg["S"].unsqueeze(0), "r": lg["rough"].view(1, 1, *lg["rough"].shape),
         "eta": torch.ones(1, 1, *lg["rough"].shape), "k": torch.ones(1, 1, *lg["rough"].shape)}
    phys = PhysicsStokesLayer("copper", "environment" if lg["mode"] == "environment" else "directional")
    psi_deg = 0.0
    with torch.no_grad():
        par = phys.chain_intensity(z, psi_deg, psi_deg, lg["eta"].unsqueeze(0), lg["kappa"].unsqueeze(0), lg["n_diff"].view(1, 1, *lg["n_diff"].shape))[0]
        crs = phys.chain_intensity(z, psi_deg, psi_deg + 90.0, lg["eta"].unsqueeze(0), lg["kappa"].unsqueeze(0), lg["n_diff"].view(1, 1, *lg["n_diff"].shape))[0]
    ref = d["lin"]
    obj = d["obj_mask"].numpy()
    # exposure matching: a real source polarizer halves the light and operators re-expose, so each virtual image is scaled to the RGB
    # median cartridge luminance BEFORE clipping; otherwise "glare suppression" would just measure a darker image.
    med = lambda x: float(np.median(_lum(x.numpy())[obj]))
    par, crs = par * med(ref) / med(par), crs * med(ref) / med(crs)
    ys, xs = np.nonzero(obj)
    sl = (slice(ys.min() - 2, ys.max() + 3), slice(xs.min() - 2, xs.max() + 3))
    bg = np.array([0.5, 0.5, 0.5])
    def show(x):                                            # background shown as neutral gray in every panel (virtual images model the object only)
        im = linear_to_srgb(x.clamp(0, 1)).permute(1, 2, 0).numpy()
        return np.where(obj[..., None], im, bg)[sl]
    sat = lambda x: float((np.clip(_lum(x.numpy()), 0, None)[obj] >= 0.95).mean())      # saturated fraction of cartridge pixels
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.2))
    for a_, (t, im) in zip(ax, [("RGB (unpolarized)", ref), ("virtual parallel", par), ("virtual crossed", crs)]):
        a_.imshow(show(im)); a_.set_title(t, fontsize=7); a_.axis("off")
    ax[3].axis("off")
    ax[3].text(0.0, 0.95, f"fraction of cartridge pixels with\nlinear luminance >= 0.95 (exposure-matched):\n  RGB:      {sat(ref):.3f}\n  parallel: {sat(par):.3f}\n  crossed:  {sat(crs):.3f}\n\nsource polarizer at {psi_deg:g} deg,\nanalyzer at {psi_deg + 90:g} deg.\nTrue latent state (upper bound).\nOPEN ISSUE: crossed-state leakage at\noblique incidence depends on the frame\nconvention linking source and analyzer\nzeros; unvalidated until the hardware\nexperiment (POLARIZATION_PHYSICS.md).", va="top", fontsize=6.3)
    provenance_note(fig, SYNTH_NOTE + " Model prediction only (renderer and Mueller chain share one Fresnel model); crossed-state result is convention-sensitive and NOT validated.")
    return save_all(fig, "fig06_rgb_vs_virtual_crosspol", outdir)


# --------------------------------------------------------------------------------------- Figure 9
def fig09_analyzer_angle_optimization(outdir="figures", tables="tables", **_):
    apply_style()
    d = _synthetic_sample(mode="environment", size=128, exposure_range=(1.2, 1.7))
    va = VirtualAnalyzer()
    angles = np.arange(0, 180, 15.0)
    st = 2.0 * va(d["S_gt"].unsqueeze(0), list(angles))[0]      # +1 stop exposure compensation (analyzer passes ~half of S0)
    lum = np.stack([_lum(np.clip(s.numpy(), 0, 1)) for s in st])
    mask = (d["mask"] > 1).numpy()
    res = optimise_analyzer(lum, angles, mask=mask, weights=(1.0, 1.0, 1.0, 1.0, 0.0))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    a = ax[0]
    for (k, c, mk) in zip(("T", "C", "E", "G"), SERIES, MARKERS):
        a.plot(angles, res.terms_norm[k], mk, color=c, label=k, ms=4)
    a.set(xlabel="analyzer angle θ (deg)", ylabel="normalised term (min–max)", title="Objective terms (computed at sampled angles)", ylim=(-0.05, 1.3)); a.legend(frameon=False, ncol=4, loc="upper center", fontsize=6.5)
    a = ax[1]
    a.plot(angles, res.J, "o", color=SERIES[0], label="J(θ), sampled")
    a.plot(res.fine_angles_deg, res.J_fit, color=SERIES[0], lw=1.2, label="periodic PCHIP (fit)")
    a.axvline(res.theta_star_deg, color=SERIES[1], ls="--", label=f"θ* = {res.theta_star_deg:.1f}°")
    a.set(xlabel="analyzer angle θ (deg)", ylabel="J(θ) = αT + βC + γE − δG", title="Analyzer angle vs objective"); a.legend(frameon=False, fontsize=6.5)
    os.makedirs(tables, exist_ok=True)
    pd.DataFrame({"sample_id": "synthetic_demo", "angle_deg": angles, **{k: res.terms[k] for k in "TCEGU"}, "J": res.J, "data_origin": "SYNTHETIC"}).to_csv(
        os.path.join(tables, "fig09_angle_objective_raw.csv"), index=False)
    pd.DataFrame({"angle_deg": res.fine_angles_deg, "J_fit": res.J_fit, "method": "periodic_pchip"}).to_csv(os.path.join(tables, "fig09_angle_objective_fitted.csv"), index=False)
    provenance_note(fig, SYNTH_NOTE + " Angles are the sampled analyzer angles; the curve is a periodic PCHIP interpolant (separate CSV), not data.")
    return save_all(fig, "fig09_analyzer_angle_optimization", outdir)


# --------------------------------------------------------------------------------------- result-driven figures
def _npz_samples(results: str, n: int = 4):
    files = sorted(glob.glob(os.path.join(results, "predictions", "*.npz")))
    if not files:
        print(f"SKIPPED: requires {results}/predictions/*.npz (see experiments.export.export_predictions).")
        return None
    return [np.load(f, allow_pickle=True) for f in files[:n]]


def fig07_corrosion_segmentation(outdir="figures", results="results", n=4, **_):
    apply_style()
    data = _npz_samples(results, n)
    if not data:
        return None
    methods = [k.split("__")[1] for k in data[0].files if k.startswith("pred__")]
    fig, ax = plt.subplots(len(data), 2 + len(methods), figsize=(1.5 * (2 + len(methods)), 1.5 * len(data)), squeeze=False)
    for i, d in enumerate(data):
        ax[i, 0].imshow(d["rgb"].transpose(1, 2, 0)); ax[i, 1].imshow(d["gt"], cmap=CLASS_CMAP, vmin=0, vmax=6, interpolation="nearest")
        for j, m in enumerate(methods):
            ax[i, 2 + j].imshow(d[f"pred__{m}"], cmap=CLASS_CMAP, vmin=0, vmax=6, interpolation="nearest")
            if i == 0:
                ax[0, 2 + j].set_title(m, fontsize=6)
        ax[0, 0].set_title("RGB", fontsize=6); ax[0, 1].set_title("annotation", fontsize=6)
        for a in ax[i]:
            a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=INK2, lw=0.3) for c in CLASS_COLORS]
    fig.legend(handles, CLASS_NAMES, ncol=7, loc="lower center", fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    provenance_note(fig, f"data origin: {str(data[0]['data_origin'])}; held-out test samples only.")
    return save_all(fig, "fig07_corrosion_segmentation", outdir)


def fig08_pitting_detection(outdir="figures", results="results", n=4, **_):
    apply_style()
    data = _npz_samples(results, n)
    if not data:
        return None
    methods = [k.split("__")[1] for k in data[0].files if k.startswith("pit__")]
    if not methods:
        print("SKIPPED: predictions contain no pit__<method> maps."); return None
    fig, ax = plt.subplots(len(data), 2 + len(methods), figsize=(1.5 * (2 + len(methods)), 1.5 * len(data)), squeeze=False)
    for i, d in enumerate(data):
        ax[i, 0].imshow(d["rgb"].transpose(1, 2, 0)); ax[i, 1].imshow(d["gt"] == 6, cmap="Greys", interpolation="nearest")
        for j, m in enumerate(methods):
            ax[i, 2 + j].imshow(d[f"pit__{m}"], cmap=SEQ, vmin=0, vmax=1)
            if i == 0:
                ax[0, 2 + j].set_title(f"{m}: P(pit)", fontsize=6)
        ax[0, 0].set_title("RGB", fontsize=6); ax[0, 1].set_title("pit annotation", fontsize=6)
        for a in ax[i]:
            a.axis("off")
    provenance_note(fig, f"data origin: {str(data[0]['data_origin'])}.")
    return save_all(fig, "fig08_pitting_detection", outdir)


def fig10_physical_vs_virtual_accuracy(outdir="figures", results="results", **_):
    apply_style()
    df = _csv(results, "physical_vs_virtual")
    if df is None:
        return None
    cols = [c for c in ("miou", "pol_dolp_mae", "pol_aolp_wrapped_err_deg", "glare_supp_virt", "latency_ms_cpu") if c in df and df[c].notna().any()]
    fig, ax = plt.subplots(1, len(cols), figsize=(1.9 * len(cols), 2.6), squeeze=False)
    cfgs = list(dict.fromkeys(df["config"]))
    for a, c in zip(ax[0], cols):
        for i, cf in enumerate(cfgs):
            v = df.loc[df["config"] == cf, c].dropna().values
            a.plot(np.full(len(v), i) + np.linspace(-0.08, 0.08, max(len(v), 1))[:len(v)], v, MARKERS[i % 4], color=SERIES[i % 4], alpha=0.8)   # per-seed points
            if len(v):
                a.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=INK, lw=1.4)                                                            # mean
        a.set_xticks(range(len(cfgs))); a.set_xticklabels([x.replace("_", "\n") for x in cfgs], fontsize=5.5); a.set_title(c, fontsize=7)
    fig.tight_layout()
    provenance_note(fig, "markers: individual seeds; bar: mean. Origin: " + ",".join(map(str, df.get("data_origin", pd.Series(["?"])).dropna().unique())))
    return save_all(fig, "fig10_physical_vs_virtual_accuracy", outdir)


def fig11_ablation(outdir="figures", results="results", **_):
    apply_style()
    df = _csv(results, "ablation")
    if df is None:
        return None
    arms = list(dict.fromkeys(df["arm"]))
    fig, ax = plt.subplots(figsize=(3.6, 0.28 * len(arms) + 1.0))
    for i, a_ in enumerate(arms):
        v = df.loc[df["arm"] == a_, "miou"].values
        ax.plot(v, np.full(len(v), i), "o", color=SERIES[0], alpha=0.55)
        m = v.mean()
        ci = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0
        ax.errorbar([m], [i], xerr=[ci], fmt="D", color=INK, ms=4, capsize=2, lw=1)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=6); ax.invert_yaxis(); ax.set(xlabel="mIoU (held-out)")
    provenance_note(fig, "blue: individual seeds; black: mean ± 95% t-interval over seeds. Origin: " + ",".join(map(str, df["data"].unique())))
    return save_all(fig, "fig11_ablation_study", outdir)


def fig12_robustness(outdir="figures", results="results", **_):
    apply_style()
    df = _csv(results, "robustness")
    if df is None:
        return None
    perts = list(dict.fromkeys(df["perturbation"]))
    nc = min(4, len(perts)); nr = math.ceil(len(perts) / nc)
    fig, ax = plt.subplots(nr, nc, figsize=(1.9 * nc, 1.8 * nr), squeeze=False)
    methods = list(dict.fromkeys(df["method"]))
    for a, p in zip(ax.ravel(), perts):
        for i, m in enumerate(methods):
            sub = df[(df.perturbation == p) & (df.method == m)]
            g = sub.groupby("level")["miou"].agg(["mean", "std"]).reset_index()
            a.plot(sub["level"], sub["miou"], MARKERS[i % 4], color=SERIES[i % 4], alpha=0.25, ms=3)         # per-seed points
            a.errorbar(g["level"], g["mean"], yerr=g["std"].fillna(0), fmt=MARKERS[i % 4] + "-", color=SERIES[i % 4], ms=4, lw=1, capsize=1.5, label=m)
        a.set_title(p, fontsize=7)
    for a in ax.ravel()[len(perts):]:
        a.axis("off")
    ax.ravel()[0].legend(frameon=False, fontsize=5.5)
    fig.supylabel("mIoU", fontsize=7)
    fig.tight_layout()
    provenance_note(fig, "faint: per-seed; solid: mean ± sd over seeds, joined through MEASURED levels (no smoothing).")
    return save_all(fig, "fig12_robustness", outdir)


def fig13_accuracy_vs_runtime(outdir="figures", results="results", **_):
    apply_style()
    seg, rt = _csv(results, "overall_segmentation"), _csv(results, "runtime")
    if seg is None or rt is None:
        return None
    rt = rt[rt["device"] == "cpu"].groupby("model")["mean_ms"].mean()
    g = seg.groupby("method")["miou"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, (m, row) in enumerate(g.iterrows()):
        if m in rt.index:
            ax.errorbar([rt[m]], [row["mean"]], yerr=[0 if np.isnan(row["std"]) else row["std"]], fmt=MARKERS[i % 4], color=SERIES[i % 4], capsize=2)
            ax.annotate(m, (rt[m], row["mean"]), textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set(xlabel="CPU latency (ms/frame)", ylabel="mIoU (mean ± sd over seeds)")
    return save_all(fig, "fig13_accuracy_vs_runtime", outdir)


def fig14_uncertainty(outdir="figures", results="results", **_):
    apply_style()
    rel, rc = _csv(results, "reliability"), _csv(results, "risk_coverage")
    if rel is None or rc is None:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax[0].plot([0, 1], [0, 1], color=INK2, ls=":", lw=1)
    for i, (m, g) in enumerate(rel.groupby("method")):
        g = g[g["count"] > 0]
        ax[0].plot(g["confidence"], g["accuracy"], MARKERS[i % 4] + "-", color=SERIES[i % 4], label=m, ms=3.5, lw=0.8)
    ax[0].set(xlabel="confidence", ylabel="accuracy", title="reliability"); ax[0].legend(frameon=False, fontsize=6)
    for i, (m, g) in enumerate(rc.groupby("method")):
        ax[1].plot(g["coverage"], g["risk"], MARKERS[i % 4] + "-", color=SERIES[i % 4], label=m, ms=3, lw=0.8)
    ax[1].set(xlabel="coverage (fraction kept)", ylabel="risk (error rate)", title="risk–coverage")
    data = _npz_samples(results, 1)
    if data:
        d = data[0]; m = [k.split("__")[1] for k in d.files if k.startswith("entropy__")][0]
        h = ax[2].imshow(d[f"entropy__{m}"], cmap=SEQ); plt.colorbar(h, ax=ax[2], fraction=0.05); ax[2].set_title(f"predictive entropy ({m})"); ax[2].axis("off")
    else:
        ax[2].axis("off")
    fig.tight_layout()
    return save_all(fig, "fig14_uncertainty", outdir)


def fig15_failure_cases(outdir="figures", results="results", n=4, **_):
    apply_style()
    data = _npz_samples(results, 64)
    if not data:
        return None
    methods = [k.split("__")[1] for k in data[0].files if k.startswith("pred__")]
    m = methods[-1]
    miou = [(float((d[f"pred__{m}"] == d["gt"]).mean()), i) for i, d in enumerate(data)]
    worst = [data[i] for _, i in sorted(miou)[:n]]
    fig, ax = plt.subplots(len(worst), 4, figsize=(6.0, 1.5 * len(worst)), squeeze=False)
    for i, d in enumerate(worst):
        ax[i, 0].imshow(d["rgb"].transpose(1, 2, 0)); ax[i, 1].imshow(d["gt"], cmap=CLASS_CMAP, vmin=0, vmax=6, interpolation="nearest")
        ax[i, 2].imshow(d[f"pred__{m}"], cmap=CLASS_CMAP, vmin=0, vmax=6, interpolation="nearest")
        ax[i, 3].imshow(d[f"entropy__{m}"], cmap=SEQ)
        for a in ax[i]:
            a.axis("off")
    for a, t in zip(ax[0], ("RGB", "annotation", f"{m}", "entropy")):
        a.set_title(t, fontsize=6)
    provenance_note(fig, "lowest pixel-accuracy test samples (selected by an automatic rule, not hand-picked).")
    return save_all(fig, "fig15_failure_cases", outdir)


REGISTRY = {2: fig02_physical_vs_virtual, 3: fig03_polarization_physics, 4: fig04_virtual_analyzer_sweep, 5: fig05_stokes_dolp_aolp,
            6: fig06_rgb_vs_virtual_crosspol, 7: fig07_corrosion_segmentation, 8: fig08_pitting_detection, 9: fig09_analyzer_angle_optimization,
            10: fig10_physical_vs_virtual_accuracy, 11: fig11_ablation, 12: fig12_robustness, 13: fig13_accuracy_vs_runtime,
            14: fig14_uncertainty, 15: fig15_failure_cases}
