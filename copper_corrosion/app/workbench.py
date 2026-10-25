"""Corrosion Segmentation Workbench - run every approach step by step and compare them.

    streamlit run app/workbench.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from cucorr.config import config_hash, load_config, resolve  # noqa: E402
from cucorr.data.splits import group_split  # noqa: E402
from cucorr.data.synthetic import synthetic_image  # noqa: E402
from cucorr.data.workbooks import audit, load_manifest  # noqa: E402
from cucorr.reporting.outputs import save_annotation_outputs  # noqa: E402
from cucorr.schema import CLASS_NAMES, NUM_CLASSES, UNCERTAIN  # noqa: E402
from cucorr.visualization.render import boundary_overlay, color_mask, overlay, uncertainty_map  # noqa: E402
from cucorr.workbench import charts as ch  # noqa: E402
from cucorr.workbench import engine as en  # noqa: E402

st.set_page_config(page_title="Corrosion Workbench", page_icon="🟢", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0d3f80 0%, #1b6f8a 45%, #1baf7a 100%); border-radius: 18px;
       padding: 28px 32px; color: #fff; margin-bottom: 18px; box-shadow: 0 8px 28px rgba(13,63,128,.18);}
.hero h1 {color:#fff; margin:0; font-size: 2.0rem; letter-spacing:-.02em;}
.hero p {color: rgba(255,255,255,.88); margin: 6px 0 0; font-size: 1.02rem;}
.steps {display:flex; gap:10px; margin: 6px 0 22px; flex-wrap: wrap;}
.step {flex:1; min-width: 130px; border-radius: 12px; padding: 10px 14px; background:#f1f0ec; color:#52514e;
       border: 1px solid #e6e5e0; font-size:.9rem;}
.step b {display:block; font-size: .78rem; text-transform: uppercase; letter-spacing:.06em; opacity:.8}
.step.done {background:#e8f7f0; border-color:#1baf7a; color:#0b5a3f;}
.step.now {background:#fff; border-color:#2a78d6; color:#0d3f80; box-shadow: 0 2px 10px rgba(42,120,214,.15);}
.card {background:#fff; border:1px solid #e6e5e0; border-radius:14px; padding:16px 18px; min-height:170px;}
.card h4 {margin:0 0 4px; font-size:1rem;}
.card p {margin:0; color:#52514e; font-size:.88rem;}
.kpi {background:#fff; border:1px solid #e6e5e0; border-radius:14px; padding:14px 18px;}
.kpi .v {font-size:1.7rem; font-weight:700; color:#0b0b0b; line-height:1.1}
.kpi .l {font-size:.8rem; color:#52514e; text-transform:uppercase; letter-spacing:.05em}
.kpi .s {font-size:.82rem; color:#52514e}
.chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; margin-right:6px;
       background:#f1f0ec; color:#52514e; border:1px solid #e6e5e0}
.warn {background:#fff7e6; border:1px solid #eda100; color:#6b4a00; border-radius:12px; padding:10px 14px; margin:6px 0 14px;}
.legend span {display:inline-flex; align-items:center; margin-right:14px; font-size:.85rem; color:#52514e}
.legend i {width:14px; height:14px; border-radius:4px; margin-right:6px; display:inline-block; border:1px solid #ccc}
div[data-testid="stTabs"] button p {font-size: 1rem; font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("items", [])
ss.setdefault("results", {})
ss.setdefault("models", {})
ss.setdefault("log", [])
ss.setdefault("selected", ["A · RGB distance", "A · CIEDE2000", "B · Random Forest", "B · Gradient Boosting",
                           "C · Superpixels", "F · Hybrid"])


@st.cache_data(show_spinner=False)
def cached_audit():
    cfg = load_config()
    return audit(resolve(cfg, "settings_xlsx"), resolve(cfg, "labels_xlsx")), load_manifest(resolve(cfg, "labels_xlsx"))


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def legend_html():
    rows = "".join(f'<span><i style="background:{ch.CLASS_COLORS[i]}"></i>{n}</span>' for i, n in enumerate(CLASS_NAMES))
    rows += f'<span><i style="background:{ch.CLASS_COLORS[-1]}"></i>Uncertain</span>'
    return f'<div class="legend">{rows}</div>'


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- sidebar: settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    cfg = load_config()
    seed = st.number_input("Random seed", value=int(cfg["seed"]))
    st.markdown("**Approach A · colour matching**")
    margin = st.slider("Abstain margin (ΔE)", 0.0, 10.0, float(cfg["approach_a"]["abstain_margin"]), 0.5)
    maxd = st.slider("Max distance (ΔE)", 5.0, 80.0, float(cfg["approach_a"]["max_distance"]), 1.0)
    use_manifest = st.toggle("Use 660 manifest centroids as prototypes", value=True)
    norm = st.selectbox("Colour normalisation", ["none", "gray_world", "max_rgb"])
    st.markdown("**Approach B / C**")
    px = st.select_slider("Training pixels per image", [1000, 2000, 4000, 8000], value=4000)
    nseg = st.select_slider("Superpixels per image", [100, 200, 400, 800], value=400)
    smooth = st.slider("Graph smoothing strength", 0.0, 1.0, 0.5, 0.1)
    st.markdown("**Evaluation**")
    robust = st.toggle("Robustness tests (7 perturbations)", value=True)
    alpha_ov = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    cfg = load_config(overrides={
        "seed": int(seed), "color": {"normalization": norm},
        "approach_a": {"abstain_margin": margin, "max_distance": maxd, "use_manifest_prototypes": use_manifest},
        "approach_b": {"pixels_per_image": px, "n_estimators": 150},
        "approach_c": {"n_segments": nseg, "smoothing": smooth},
        "approach_d": {"epochs": 20, "mc_dropout_samples": 4},
        "evaluation": {"bootstrap_iterations": 1000}, "outputs": {"overlay_alpha": alpha_ov}})
    st.caption(f"config hash `{config_hash(cfg)}`")
    if st.button("🗑️ Reset session", width="stretch"):
        for k in ("items", "results", "models", "log"):
            ss[k] = [] if k in ("items", "log") else {}
        st.rerun()

# ---------------------------------------------------------------- header + stepper
items: list[en.Item] = ss["items"]
results: dict[str, en.RunResult] = ss["results"]
has_ref = any(i.ref is not None for i in items)
done = [True, bool(items), bool(items) and bool(ss["selected"]), bool(results), bool(results), bool(results)]
st.markdown('<div class="hero"><h1>Copper Corrosion Segmentation Workbench</h1>'
            '<p>Load images → choose approaches → run step by step → inspect masks → compare with statistics.</p></div>',
            unsafe_allow_html=True)
labels = ["Inputs audit", "Load images", "Choose approaches", "Run", "Inspect results", "Compare"]
cur = next((i for i, d in enumerate(done) if not d), 5)
st.markdown('<div class="steps">' + "".join(
    f'<div class="step {"done" if done[i] and i != cur else ("now" if i == cur else "")}"><b>Step {i + 1}</b>{l}</div>'
    for i, l in enumerate(labels)) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["① Inputs", "② Images", "③ Approaches", "④ Run", "⑤ Results", "⑥ Compare", "⑦ Export"])

# ---------------------------------------------------------------- 1. inputs
with tabs[0]:
    rep, manifest = cached_audit()
    c = st.columns(4)
    kpi(c[0], "Classes", "6", "Healthy + 5 corrosion products")
    kpi(c[1], "Reference colours", "24", "4 per class (settings.xlsx)")
    kpi(c[2], "Labelled clusters", f"{rep['manifest']['rows']}", "CIELAB centroids (labels.xlsx)")
    kpi(c[3], "Cross-file agreement", f"{rep['consistency']['manifest_vs_settings_nearest_ref_agreement']:.0%}",
        "nearest settings colour = manifest label")
    st.write("")
    l, r = st.columns([1, 1.25])
    with l:
        st.markdown("#### Expert reference colours")
        st.plotly_chart(ch.swatch_figure(rep["classes"]), width="stretch", config={"displayModeBar": False})
    with r:
        st.plotly_chart(ch.manifest_scatter(manifest), width="stretch", config={"displayModeBar": False})
    st.markdown("#### Audit warnings")
    for w in rep["warnings"]:
        st.markdown(f'<div class="warn">⚠️ {w}</div>', unsafe_allow_html=True)
    with st.expander("Assumptions and unresolved questions"):
        st.markdown("\n".join(f"- {a}" for a in rep["assumptions"] + rep["unresolved"]))

# ---------------------------------------------------------------- 2. images
with tabs[1]:
    st.markdown("#### Add images")
    src1, src2, src3 = st.columns(3)
    with src1:
        st.markdown('<div class="card"><h4>📤 Upload</h4><p>Real photographs. Optional masks with the same file '
                    'name (PNG, values 0–5, 255 = uncertain) enable accuracy metrics.</p></div>', unsafe_allow_html=True)
        ups = st.file_uploader("Images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True)
        mups = st.file_uploader("Reference masks (optional)", type=["png"], accept_multiple_files=True)
        max_side = st.select_slider("Downscale longest side to", [256, 384, 512, 768, 1024, 0], value=512,
                                    format_func=lambda v: "original" if v == 0 else f"{v}px")
        if st.button("Add uploaded", type="primary", disabled=not ups):
            masks = {Path(m.name).stem: Image.open(m) for m in (mups or [])}
            for u in ups:
                im = Image.open(u).convert("RGB")
                mk = masks.get(Path(u.name).stem)
                if max_side and max(im.size) > max_side:
                    s = max_side / max(im.size)
                    size = (round(im.width * s), round(im.height * s))
                    im = im.resize(size, Image.LANCZOS)
                    mk = mk.resize(size, Image.NEAREST) if mk is not None else None
                ref = np.asarray(mk).astype(np.uint8) if mk is not None else None
                if ref is not None and ref.ndim != 2:
                    st.error(f"{u.name}: mask must be single-channel"); continue
                items.append(en.Item(Path(u.name).stem, np.asarray(im, np.uint8), ref, "test"))
            st.rerun()
    with src2:
        st.markdown('<div class="card"><h4>🧪 Synthetic demo set</h4><p>Copper-like images with exact masks, '
                    'for checking that the pipeline runs. Its results are not corrosion results.</p></div>',
                    unsafe_allow_html=True)
        n = st.slider("Number of images", 8, 60, 24, 4)
        if st.button("Generate synthetic set"):
            for i in range(n):
                img, m = synthetic_image(128, 160, seed=1000 + i)
                items.append(en.Item(f"syn_{i:03d}", img, m, "test"))
            ss["specimen"] = {f"syn_{i:03d}": f"spec_{i // 2}" for i in range(n)}
            st.rerun()
    with src3:
        st.markdown('<div class="card"><h4>📁 Project folder</h4><p>Load <code>data/test_images</code> (and '
                    'masks from <code>data/test_masks</code> if present).</p></div>', unsafe_allow_html=True)
        folder = st.text_input("Folder", "data/test_images")
        if st.button("Load folder"):
            from cucorr.utils.io import list_images, read_mask, read_rgb
            fp = ROOT / folder
            for p in list_images(fp):
                mp = ROOT / "data" / "test_masks" / f"{p.stem}.png"
                items.append(en.Item(p.stem, read_rgb(p), read_mask(mp) if mp.exists() else None, "test"))
            st.rerun()

    if items:
        st.divider()
        nref = sum(i.ref is not None for i in items)
        c = st.columns(4)
        kpi(c[0], "Images", len(items))
        kpi(c[1], "With reference mask", nref)
        kpi(c[2], "Prediction-only", len(items) - nref, "no accuracy, comparison by agreement")
        if nref >= 4:
            frac = st.slider("Share of labelled specimens used for training", 0.3, 0.8, 0.6, 0.05)
            lab = [i for i in items if i.ref is not None]
            groups = [ss.get("specimen", {}).get(i.image_id, i.image_id) for i in lab]
            sp = group_split([i.image_id for i in lab], groups, (frac, 0.0, 1 - frac), seed=int(seed))
            sp.loc[sp.split == "val", "split"] = "test"
            smap = dict(zip(sp.image_id, sp.split))
            for i in items:
                i.split = smap.get(i.image_id, "test")
        ntr = sum(i.split == "train" for i in items)
        kpi(c[3], "Train / test", f"{ntr} / {len(items) - ntr}", "split by specimen, never by pixel")
        if not has_ref:
            st.markdown('<div class="warn">No reference masks were provided, so only approach A (which needs no '
                        'training) can run, and the comparison is about agreement between methods, not '
                        'accuracy.</div>', unsafe_allow_html=True)
        st.write("")
        cols = st.columns(6)
        for k, it in enumerate(items[:18]):
            with cols[k % 6]:
                st.image(it.image, caption=f"{it.image_id} · {it.split}{' · mask' if it.ref is not None else ''}",
                         width="stretch")
        if len(items) > 18:
            st.caption(f"+ {len(items) - 18} more")

# ---------------------------------------------------------------- 3. approaches
with tabs[2]:
    st.markdown("#### Choose the approaches to run")
    sel = []
    cols = st.columns(3)
    for k, (key, spec) in enumerate(en.CATALOG.items()):
        ok, why = en.available(key)
        needs = spec["base"] not in ("approach_a",)
        blocked = (not ok) or (needs and not any(i.split == "train" for i in items) and bool(items))
        with cols[k % 3]:
            st.markdown(f'<div class="card"><h4><span style="color:{ch.color_of(key)}">●</span> {key}</h4>'
                        f'<p>{spec["desc"]}</p><p style="margin-top:6px">'
                        f'<span class="chip">{"needs training masks" if needs else "no training"}</span>'
                        f'{"<span class=chip>⚠ " + why + "</span>" if why else ""}</p></div>', unsafe_allow_html=True)
            if st.checkbox("Include", value=(key in ss["selected"]) and not blocked, key=f"sel_{key}",
                           disabled=blocked):
                sel.append(key)
            st.write("")
    ss["selected"] = sel or ss["selected"]

# ---------------------------------------------------------------- 4. run
with tabs[3]:
    sel = [k for k in ss["selected"] if st.session_state.get(f"sel_{k}", False)]
    train = [i for i in items if i.split == "train" and i.ref is not None]
    test = [i for i in items if i.split == "test"]
    rare = en.rare_classes(train)
    st.markdown("#### Run pipeline")
    if not items:
        st.info("Add images in step ②.")
    elif not sel:
        st.info("Choose at least one approach in step ③.")
    else:
        pending = [k for k in sel if k not in results]
        c = st.columns([1, 1, 1, 2])
        run_next = c[0].button("▶ Run next approach", type="primary", disabled=not pending, width="stretch")
        run_all = c[1].button("⏩ Run all remaining", disabled=not pending, width="stretch")
        if c[2].button("↺ Clear results", width="stretch"):
            ss["results"], ss["models"], ss["log"] = {}, {}, []
            st.rerun()
        c[3].markdown(" ".join(f'<span class="chip" style="border-color:{ch.color_of(k)}">'
                               f'{"✅" if k in results else "⏳"} {k}</span>' for k in sel), unsafe_allow_html=True)
        todo = pending[:1] if run_next else (pending if run_all else [])
        for key in todo:
            res = en.RunResult(key)
            with st.status(f"Running **{key}**", expanded=True) as status:
                try:
                    st.write(f"**Step 1 · fit** on {len(train)} training images" if en.make(key, cfg).requires_training
                             else "**Step 1 · fit** - not needed (expert reference colours)")
                    seg, res.fit_seconds = en.fit(key, cfg, train)
                    res.describe = seg.describe()
                    st.write(f"✓ fitted in {res.fit_seconds:.1f}s")
                    st.write(f"**Step 2 · predict** {len(test)} images")
                    bar = st.progress(0.0)
                    for j, it in enumerate(test):
                        res.preds[it.image_id], res.seconds[it.image_id] = en.predict(seg, it)
                        bar.progress((j + 1) / len(test), text=f"{it.image_id}")
                    if any(t.ref is not None for t in test):
                        st.write("**Step 3 · evaluate** against reference masks")
                        en.evaluate(res, seg, test, cfg, rare)
                        if robust:
                            st.write("**Step 4 · robustness** - brightness, white balance, gamma, shadow, glare, noise")
                            en.robustness(res, seg, test, int(seed))
                    else:
                        st.write("**Step 3 · evaluate** - skipped (no reference masks)")
                    results[key] = res
                    ss["models"][key] = seg
                    ss["log"].append(f"{time.strftime('%H:%M:%S')} {key}: done")
                    status.update(label=f"✅ {key} finished", state="complete", expanded=False)
                except Exception as e:  # surface errors in the UI rather than crashing the app
                    res.error = str(e)
                    status.update(label=f"❌ {key} failed: {e}", state="error")
        if todo:
            st.rerun()
        if results:
            st.markdown("#### Completed")
            rows = []
            for k, r in results.items():
                s = en.summarize(r, cfg)
                rows.append({"Approach": k, "Images": len(r.preds), "Fit (s)": round(r.fit_seconds, 2),
                             "s / image": round(s["seconds_per_image"], 3), "Uncertain px": f"{s['review_fraction']:.1%}",
                             "mIoU": fmt(s["macro_miou"]) if r.metrics else "—"})
            st.dataframe(rows, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. results gallery
with tabs[4]:
    if not results:
        st.info("Run at least one approach in step ④.")
    else:
        ids = list(next(iter(results.values())).preds)
        c = st.columns([2, 2, 1])
        iid = c[0].selectbox("Image", ids)
        view = c[1].radio("View", ["Overlay", "Class mask", "Boundaries", "Uncertainty"], horizontal=True)
        hide = c[2].multiselect("Hide classes", list(CLASS_NAMES))
        hidden = {CLASS_NAMES.index(h) for h in hide}
        it = next(i for i in items if i.image_id == iid)
        st.markdown(legend_html(), unsafe_allow_html=True)
        panels = [("Original", it.image)]
        if it.ref is not None:
            panels.append(("Reference", overlay(it.image, it.ref, alpha_ov, hidden) if view == "Overlay"
                           else color_mask(it.ref, hidden)))
        for k, r in results.items():
            if iid not in r.preds:
                continue
            m, u, _ = r.preds[iid]
            img = {"Overlay": lambda: overlay(it.image, m, alpha_ov, hidden), "Class mask": lambda: color_mask(m, hidden),
                   "Boundaries": lambda: boundary_overlay(it.image, m),
                   "Uncertainty": lambda: uncertainty_map(u)}[view]()
            cap = k
            if iid in r.metrics:
                cap += f" · mIoU {r.metrics[iid]['macro_miou']:.3f}"
            panels.append((cap, img))
        cols = st.columns(min(4, len(panels)))
        for j, (cap, img) in enumerate(panels):
            cols[j % len(cols)].image(img, caption=cap, width="stretch", clamp=True)

        st.markdown("#### Per-image statistics")
        rows = []
        for k, r in results.items():
            if iid not in r.preds:
                continue
            m, u, p = r.preds[iid]
            row = {"Approach": k}
            for cidx, n in enumerate(CLASS_NAMES):
                row[n] = f"{(m == cidx).mean():.1%}"
            row["Uncertain"] = f"{(m == UNCERTAIN).mean():.1%}"
            row["Corroded"] = f"{np.isin(m, [1, 2, 3, 4, 5]).mean():.1%}"
            row["Mean confidence"] = f"{p.max(-1).mean():.2f}"
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Fractions are of image pixels; physical areas need a px/mm calibration.")

# ---------------------------------------------------------------- 6. compare
with tabs[5]:
    if not results:
        st.info("Run approaches in step ④ to compare them.")
    else:
        scored = {k: r for k, r in results.items() if r.metrics}
        summ = {k: en.summarize(r, cfg) for k, r in scored.items()}
        if scored:
            synthetic = all(i.image_id.startswith("syn_") for i in items if i.ref is not None)
            if synthetic:
                st.markdown('<div class="warn">These reference masks are <b>synthetic</b>. The numbers check that the '
                            'pipeline runs and do not measure corrosion-segmentation accuracy.</div>',
                            unsafe_allow_html=True)
            rk = en.ranking(results, cfg)
            best = rk[0][0]
            stats = en.statistics(results, cfg)
            c = st.columns(4)
            kpi(c[0], "Top-ranked (predeclared rule)", best, f"score {rk[0][1]:.2f}")
            kpi(c[1], "Best mIoU", fmt(max(s["macro_miou"] for s in summ.values())),
                max(summ, key=lambda k: np.nan_to_num(summ[k]["macro_miou"], nan=-1)))
            kpi(c[2], "Best boundary F1", fmt(max(np.nan_to_num(s["boundary_f1"], nan=-1) for s in summ.values())),
                max(summ, key=lambda k: np.nan_to_num(summ[k]["boundary_f1"], nan=-1)))
            kpi(c[3], "Fastest", f"{min(s['seconds_per_image'] for s in summ.values()):.2f}s",
                min(summ, key=lambda k: summ[k]["seconds_per_image"]))
            st.write("")
            sub = st.tabs(["Overview", "Per class", "Confusion", "Robustness", "Confidence", "Statistics"])
            cfgp = {"displaylogo": False}
            with sub[0]:
                l, r = st.columns(2)
                metric = l.selectbox("Metric", [("macro_miou", "Macro mIoU"), ("binary_dice", "Binary corrosion Dice"),
                                                ("boundary_f1", "Boundary F1"), ("rare_class_iou", "Rare-class IoU"),
                                                ("balanced_accuracy", "Balanced accuracy")], format_func=lambda t: t[1])
                l.plotly_chart(ch.metric_bars(summ, *metric), width="stretch", config=cfgp)
                r.plotly_chart(ch.speed_accuracy(summ), width="stretch", config=cfgp)
                st.plotly_chart(ch.metric_heatmap(summ, [("macro_miou", "mIoU"), ("corrosion_miou", "Corrosion mIoU"),
                                                         ("healthy_iou", "Healthy IoU"), ("binary_dice", "Binary Dice"),
                                                         ("boundary_f1", "Boundary F1"), ("rare_class_iou", "Rare IoU"),
                                                         ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                                                         ("small_region_recall", "Small regions"),
                                                         ("robustness_miou", "Robust mIoU")]),
                                width="stretch", config=cfgp)
            with sub[1]:
                pc = {k: en.pooled(r)["per_class_iou"] for k, r in scored.items()}
                st.plotly_chart(ch.per_class_heatmap(pc), width="stretch", config=cfgp)
                st.caption("Rare classes (from training frequencies): " +
                           ", ".join(CLASS_NAMES[c] for c in en.rare_classes(
                               [i for i in items if i.split == "train" and i.ref is not None])))
            with sub[2]:
                k = st.selectbox("Approach", list(scored), key="cm_sel")
                st.plotly_chart(ch.confusion_heatmap(en.pooled(scored[k])["confusion"], k), width="stretch", config=cfgp)
            with sub[3]:
                rob = {k: r.robustness for k, r in scored.items()}
                if any(rob.values()):
                    clean = {k: en.pooled(r)["macro_miou"] for k, r in scored.items()}
                    st.plotly_chart(ch.robustness_heatmap(rob, clean), width="stretch", config=cfgp)
                else:
                    st.info("Enable robustness tests in the sidebar and rerun.")
            with sub[4]:
                curves = {k: np.mean([m["coverage_risk"] for m in r.metrics.values()], 0) for k, r in scored.items()}
                st.plotly_chart(ch.coverage_risk_lines(curves), width="stretch", config=cfgp)
                st.caption("Curves that stay low as coverage grows have confidence scores that rank errors well. "
                           "Approach A's scores are distance-based and not calibrated probabilities, so ECE is not "
                           "reported for it.")
            with sub[5]:
                if stats:
                    rows = [{"Pair": p, "Δ mIoU": fmt(v["mean_diff"]), "95% CI": f"[{fmt(v['ci'][0])}, {fmt(v['ci'][1])}]",
                             "p (Holm)": fmt(v["p_holm"]), "Effect d_z": fmt(v["effect_size_dz"], 2), "n": v["n_images"]}
                            for p, v in stats["pairwise"].items()]
                    st.dataframe(rows, hide_index=True, width="stretch")
                    if stats["power_note"]:
                        st.markdown(f'<div class="warn">{stats["power_note"]}</div>', unsafe_allow_html=True)
                st.markdown("**Predeclared weighted-rank selection**")
                st.dataframe([{"Rank": i + 1, "Approach": k, "Score": round(s, 3)} for i, (k, s) in enumerate(rk)],
                             hide_index=True, width="stretch")
                st.caption("Weights: mIoU 0.30, binary Dice 0.15, boundary F1 0.15, rare-class IoU 0.15, ECE 0.05, "
                           "review fraction 0.10, speed 0.05, robustness 0.05. A top rank is not a universal winner.")
        # prediction-only comparison (always available)
        st.markdown("#### Comparison without reference masks")
        comp = {k: en.composition(r) for k, r in results.items() if r.preds}
        l, r = st.columns(2)
        l.plotly_chart(ch.composition_bars(comp), width="stretch", config={"displaylogo": False})
        keys, A = en.pairwise_agreement(results)
        if len(keys) > 1:
            r.plotly_chart(ch.agreement_heatmap(keys, A), width="stretch", config={"displaylogo": False})
        else:
            r.info("Run at least two approaches to see agreement between them.")

# ---------------------------------------------------------------- 7. export
with tabs[6]:
    if not results:
        st.info("Nothing to export yet.")
    else:
        st.markdown("#### Download results")
        st.write("The ZIP holds every artefact per approach and image (masks, overlays, uncertainty, stats.json, "
                 "metadata), plus a summary JSON.")
        if st.button("📦 Build ZIP", type="primary"):
            buf = io.BytesIO()
            with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                byid = {i.image_id: i for i in items}
                for k, r in results.items():
                    for iid, (m, u, p) in r.preds.items():
                        d = save_annotation_outputs(Path(td) / k.replace(" · ", "_").replace(" ", "_") / iid,
                                                    byid[iid].image, m, u, p, image_id=iid, approach=r.describe,
                                                    cfg=cfg, save_proba=False,
                                                    evaluation=({kk: vv for kk, vv in r.metrics[iid].items()
                                                                 if kk not in ("confusion", "confusion_full",
                                                                               "coverage_risk")}
                                                                if iid in r.metrics else None))
                        for f in d.iterdir():
                            zf.write(f, f.relative_to(td))
                summary = {k: {"summary": en.summarize(r, cfg), "describe": r.describe} for k, r in results.items()}
                zf.writestr("summary.json", json.dumps(summary, indent=1, default=float))
            st.download_button("⬇️ Download results.zip", buf.getvalue(), "corrosion_results.zip", "application/zip")
        st.markdown("#### Log")
        st.code("\n".join(ss["log"]) or "-")
