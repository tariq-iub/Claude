"""Streamlit annotation, review and adjudication app.

Run:  streamlit run app/annotation_app.py -- --images data/images --store data/annotations/store

Modes
* Independent labelling (blinded): no model predictions or other annotators' labels shown;
  image order randomised per annotator (seeded by annotator id).
* Review: compare proposals from several approaches, adjust opacity/class layers, view
  uncertainty, accept / reject / correct with brush, eraser, point-fill and region (superpixel) tools.
* Adjudication: shows independent labels + disagreement map; unanimous pixels pre-filled.

Every save is an append-only revision in AnnotationStore (proposals are never overwritten).
Brush/eraser require ``streamlit-drawable-canvas``; without it, point/region tools still work.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from cucorr.annotation.edits import apply_brush, flood_fill, region_fill  # noqa: E402
from cucorr.annotation.store import AnnotationStore, consensus_proposal  # noqa: E402
from cucorr.config import load_config  # noqa: E402
from cucorr.evaluation.agreement import disagreement_map  # noqa: E402
from cucorr.schema import CLASS_NAMES, UNCERTAIN  # noqa: E402
from cucorr.segmentation.superpixel import superpixels  # noqa: E402
from cucorr.utils.io import list_images, read_mask, read_rgb  # noqa: E402
from cucorr.visualization.render import color_mask, overlay, uncertainty_map  # noqa: E402

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # optional
    st_canvas = None

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images")
ap.add_argument("--store", default="data/annotations/store")
ap.add_argument("--predictions", default="outputs", help="folder with <approach>/<image_id>/ outputs")
args, _ = ap.parse_known_args()
cfg = load_config()
store = AnnotationStore(args.store)

st.set_page_config(page_title="Copper corrosion annotation", layout="wide")
ss = st.session_state
annotator = st.sidebar.text_input("Annotator ID", value=ss.get("annotator", ""))
mode = st.sidebar.radio("Mode", ["Independent (blinded)", "Review", "Adjudication"])
images = list_images(args.images)
if not annotator or not images:
    st.info("Enter an annotator ID and make sure the images folder contains images.")
    st.stop()
ss.annotator = annotator
order = np.random.default_rng(abs(hash(annotator)) % 2**32).permutation(len(images)) \
    if mode.startswith("Independent") else np.arange(len(images))
idx = st.sidebar.number_input("Image #", 0, len(images) - 1, 0)
path = images[order[idx]]
iid = path.stem
img = read_rgb(path)

key = f"{mode}:{iid}"
if ss.get("key") != key:
    ss.key = key
    ss.t0 = time.time()
    init = np.full(img.shape[:2], UNCERTAIN, np.uint8)
    ss.history, ss.future = [init], []
    ss.proposals = {}
    if mode == "Review":
        for d in sorted(Path(args.predictions).glob(f"*/{iid}/multiclass_mask.png")):
            ss.proposals[d.parent.parent.name] = read_mask(d)
        if ss.proposals:
            ss.history = [next(iter(ss.proposals.values())).copy()]
    elif mode == "Adjudication":
        labs = store.independent_labels(iid)
        ss.labels = labs
        if len(labs) >= 2:
            ss.history = [consensus_proposal(labs)[0]]

mask = ss.history[-1]
st.sidebar.markdown("### Tools")
cls_name = st.sidebar.selectbox("Class", list(CLASS_NAMES) + ["Unknown/Uncertain"])
cls = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else UNCERTAIN
tool = st.sidebar.radio("Tool", ["Brush", "Eraser (-> Uncertain)", "Point fill", "Region (superpixel)"])
radius = st.sidebar.slider("Brush radius / fill tolerance", 1, 40, 8)
alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, float(cfg["outputs"]["overlay_alpha"]))
hidden = set(st.sidebar.multiselect("Hide class layers", list(CLASS_NAMES) + ["Unknown/Uncertain"]))
hidden_ids = {CLASS_NAMES.index(h) if h in CLASS_NAMES else UNCERTAIN for h in hidden}


def push(new):
    ss.history.append(new)
    ss.future = []


c1, c2, c3 = st.columns(3)
c1.image(img, caption=f"{iid} (original)", use_container_width=True)
c2.image(color_mask(mask, hidden_ids), caption="current mask", use_container_width=True)
ov = overlay(img, mask, alpha, hidden_ids)

if st_canvas is not None and tool in ("Brush", "Eraser (-> Uncertain)"):
    with c3:
        from PIL import Image
        res = st_canvas(background_image=Image.fromarray(ov), stroke_width=2 * radius, stroke_color="#ff00ff",
                        height=img.shape[0], width=img.shape[1], drawing_mode="freedraw", key=f"cv{key}{len(ss.history)}")
    if res.image_data is not None and st.button("Apply stroke"):
        painted = res.image_data[..., 3] > 0
        push(apply_brush(mask, painted, UNCERTAIN if tool.startswith("Eraser") else cls))
        st.rerun()
else:
    c3.image(ov, caption="overlay", use_container_width=True)
    cy, cx = st.columns(2)
    y = cy.number_input("y", 0, img.shape[0] - 1, img.shape[0] // 2)
    x = cx.number_input("x", 0, img.shape[1] - 1, img.shape[1] // 2)
    if st.button("Apply at (y, x)"):
        if tool == "Point fill":
            push(flood_fill(mask, img, (y, x), cls, radius))
        elif tool == "Region (superpixel)":
            push(region_fill(mask, superpixels(img, 600, 10), (y, x), cls))
        else:
            disk = np.zeros(mask.shape, bool)
            yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
            disk[(yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2] = True
            push(apply_brush(mask, disk, UNCERTAIN if tool.startswith("Eraser") else cls))
        st.rerun()

u1, u2 = st.columns(2)
if u1.button("Undo") and len(ss.history) > 1:
    ss.future.append(ss.history.pop()); st.rerun()
if u2.button("Redo") and ss.future:
    ss.history.append(ss.future.pop()); st.rerun()

if mode == "Review" and ss.proposals:
    st.subheader("Algorithm proposals")
    cols = st.columns(len(ss.proposals))
    for col, (name, pm) in zip(cols, ss.proposals.items()):
        col.image(overlay(img, pm, alpha, hidden_ids), caption=name, use_container_width=True)
        up = Path(args.predictions) / name / iid / "uncertainty_map.png"
        if up.exists():
            col.image(read_mask(up), caption=f"{name} uncertainty", use_container_width=True, clamp=True)
        if col.button(f"Start from {name}"):
            push(pm.copy()); st.rerun()
if mode == "Adjudication":
    labs = ss.get("labels", {})
    if len(labs) < 2:
        st.warning("Needs >= 2 blinded independent annotations.")
    else:
        st.subheader("Independent labels and disagreement")
        cols = st.columns(len(labs) + 1)
        for col, (a, m) in zip(cols, labs.items()):
            col.image(color_mask(m), caption=a, use_container_width=True)
        cols[-1].image(uncertainty_map(disagreement_map(list(labs.values()))), caption="disagreement",
                       use_container_width=True, clamp=True)

note = st.text_area("Note / rationale")
elapsed = time.time() - ss.t0
b1, b2, b3, b4 = st.columns(4)
if mode.startswith("Independent"):
    if b1.button("Save independent label"):
        store.save_revision(iid, annotator, mask, "independent", note=note, seconds_spent=elapsed, blinded=True)
        st.success("saved")
elif mode == "Review":
    src = next(iter(ss.proposals), None)
    if b1.button("Accept"):
        store.save_revision(iid, annotator, mask, "accepted", src, note, elapsed); st.success("accepted")
    if b2.button("Save correction"):
        store.save_revision(iid, annotator, mask, "corrected", src, note, elapsed); st.success("saved")
    if b3.button("Reject"):
        store.save_revision(iid, annotator, mask, "rejected", src, note, elapsed); st.warning("rejected")
else:
    if b1.button("Save adjudicated decision"):
        if not note.strip():
            st.error("A rationale is required for adjudication.")
        else:
            store.adjudicate(iid, annotator, mask, note); st.success("adjudicated")
st.caption(f"Unknown/Uncertain pixels: {(mask == UNCERTAIN).mean():.1%} | time on image: {elapsed:.0f}s")
