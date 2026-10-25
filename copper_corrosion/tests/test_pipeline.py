import json

import numpy as np

from cucorr.annotation.store import AnnotationStore, consensus_proposal
from cucorr.active_learning.loop import run_active_learning
from cucorr.data.splits import check_no_leakage, group_split
from cucorr.evaluation.benchmark import run_benchmark
from cucorr.reporting.outputs import save_annotation_outputs
from cucorr.schema import UNCERTAIN
from cucorr.segmentation.registry import build
from cucorr.utils.io import read_mask, read_rgb


def test_outputs(tmp_path, cfg, dataset):
    iid, img, ref, _ = dataset[0]
    seg = build("approach_a", cfg)
    mask, unc, proba = seg.predict_mask(img)
    d = save_annotation_outputs(tmp_path / iid, img, mask, unc, proba, image_id=iid, approach=seg.describe(), cfg=cfg)
    for f in ["binary_mask.png", "color_mask.png", "multiclass_mask.png", "overlay.png", "boundary_overlay.png",
              "uncertainty_map.png", "stats.json", "annotation_metadata.json", "probabilities.npy"]:
        assert (d / f).exists(), f
    mm = read_mask(d / "multiclass_mask.png")
    assert (mm == mask).all() and read_rgb(d / "overlay.png").shape == img.shape
    b = read_mask(d / "binary_mask.png")
    assert ((b == UNCERTAIN) == (mask == UNCERTAIN)).all()
    s = json.loads((d / "stats.json").read_text())
    assert s["valid_pixels"] + s["uncertain_pixels"] == s["total_pixels"]
    assert abs(sum(c["pixels"] for c in s["classes"].values()) - s["valid_pixels"]) == 0


def test_splits_group_level():
    df = group_split([f"i{k}" for k in range(20)], [f"g{k // 4}" for k in range(20)], seed=3)
    check_no_leakage(df)
    assert set(df.split) == {"train", "val", "test"}


def test_store_append_only(tmp_path):
    st = AnnotationStore(tmp_path)
    a = np.zeros((5, 5), np.uint8); b = a.copy(); b[0] = 1
    st.save_revision("x", "ann1", a, "independent", blinded=True)
    st.save_revision("x", "ann1", b, "corrected")
    st.save_revision("x", "ann2", b, "independent", blinded=True)
    assert (st.latest("x", "ann1") == b).all()
    lab = st.independent_labels("x")
    assert (lab["ann1"] == a).all()
    prop, need = consensus_proposal(lab)
    assert (prop[0] == UNCERTAIN).all() and need[0].all() and not need[1:].any()
    st.adjudicate("x", "expert", b, "edge row is Cu2O")
    assert len((tmp_path / "audit_log.jsonl").read_text().splitlines()) == 4


def test_benchmark_end_to_end(tmp_path, cfg, dataset):
    data = [(i, im, m) for i, im, m, _ in dataset]
    sp = group_split([d[0] for d in dataset], [d[3] for d in dataset], seed=0)
    rep = run_benchmark({"A": lambda: build("approach_a", cfg), "B": lambda: build("approach_b", cfg)},
                        data, dict(zip(sp.image_id, sp.split)), cfg, tmp_path, reference_type="synthetic",
                        write_outputs=True, robustness=True)
    assert rep["provisional"] and set(rep["approaches"]) == {"A", "B"}
    assert (tmp_path / "benchmark.md").read_text().count("PROVISIONAL") == 1
    assert "A vs B" in rep["statistics"]["pairwise"]


def test_active_learning(cfg, dataset):
    pool = dataset[:6]
    r = run_active_learning(lambda: build("approach_b", cfg), [p[1] for p in pool], lambda i: pool[i][2],
                            [dataset[7][1]], [dataset[7][2]], strategy="uncertainty", budget_per_round=1,
                            max_rounds=2, init_idx=[0, 1])
    assert [c[0] for c in r["curve"]] == [2, 3, 4][:len(r["curve"])]


def test_edits(dataset):
    from cucorr.annotation.edits import apply_brush, flood_fill
    _, img, ref, _ = dataset[0]
    m = np.full(ref.shape, UNCERTAIN, np.uint8)
    out = flood_fill(m, img, (0, 0), 0, tolerance=15)
    assert (out == 0).sum() > 0 and (m == UNCERTAIN).all()  # original untouched
    assert (apply_brush(m, np.ones_like(m, bool), 3) == 3).all()
