import numpy as np
import pytest

from cucorr.schema import NUM_CLASSES, UNCERTAIN
from cucorr.segmentation.registry import build


@pytest.mark.parametrize("name", ["approach_a", "approach_b", "approach_c", "approach_f"])
def test_contract(name, cfg, dataset):
    seg = build(name, cfg)
    tr = dataset[:5]
    if seg.requires_training:
        seg.fit([d[1] for d in tr], [d[2] for d in tr])
    img = dataset[6][1]
    mask, unc, proba = seg.predict_mask(img)
    assert mask.shape == img.shape[:2] and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= set(range(NUM_CLASSES)) | {UNCERTAIN}
    assert proba.shape == (*img.shape[:2], NUM_CLASSES)
    assert np.allclose(proba.sum(-1), 1, atol=1e-3)
    assert unc.min() >= 0 and unc.max() <= 1


def test_trained_models_learn(cfg, dataset):
    from cucorr.evaluation.metrics import evaluate_segmentation
    seg = build("approach_b", cfg).fit([d[1] for d in dataset[:6]], [d[2] for d in dataset[:6]])
    _, img, ref, _ = dataset[7]
    assert evaluate_segmentation(ref, seg.predict_mask(img)[0])["pixel_accuracy"] > 0.8


def test_color_match_abstains_on_ambiguous(cfg):
    seg = build("approach_a", cfg)
    img = np.full((4, 4, 3), (255, 0, 255), np.uint8)  # saturated magenta - far from every reference
    mask, _, _ = seg.predict_mask(img)
    assert (mask == UNCERTAIN).all()


def test_rgb_baseline(cfg):
    from cucorr.config import deep_merge
    seg = build("approach_a", deep_merge(cfg, {"approach_a": {"color_space": "rgb", "distance": "euclidean",
                                                              "max_distance": 1e9, "abstain_margin": 0}}))
    img = np.full((2, 2, 3), (148, 106, 53), np.uint8)
    assert (seg.predict_mask(img)[0] == 0).all()


def test_foundation_adapter_with_classical_proposer(cfg, dataset):
    from cucorr.segmentation.foundation import FoundationSegmenter
    seg = FoundationSegmenter(cfg, build("approach_a", cfg), region_proposer="felzenszwalb")
    assert not seg.describe()["is_foundation_model"]
    m, _, p = seg.predict_mask(dataset[0][1])
    assert m.shape == dataset[0][2].shape


def test_deep_smoke(cfg, dataset):
    pytest.importorskip("torch")
    seg = build("approach_d", cfg)
    seg.fit([d[1] for d in dataset[:3]], [d[2] for d in dataset[:3]])
    m, u, p = seg.predict_mask(dataset[4][1])
    assert m.shape == dataset[4][2].shape and np.allclose(p.sum(-1), 1, atol=1e-3)


def test_lut_quantisation_matches_exact(cfg, dataset):
    from cucorr.config import deep_merge
    img = dataset[0][1]
    exact = build("approach_a", deep_merge(cfg, {"approach_a": {"lut_bits": 8}})).predict_mask(img)[0]
    quant = build("approach_a", cfg).predict_mask(img)[0]
    assert (exact == quant).mean() > 0.97
