"""Smoke test on real images placed in data/test_images/ (skipped when empty).
No reference masks exist for them, so this checks the output contract only - not accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest

from cucorr.reporting.outputs import save_annotation_outputs
from cucorr.segmentation.registry import build
from cucorr.utils.io import list_images, read_mask, read_rgb

IMAGES = list_images(Path(__file__).resolve().parents[1] / "data" / "test_images")


@pytest.mark.skipif(not IMAGES, reason="no images in data/test_images/")
@pytest.mark.parametrize("path", IMAGES, ids=lambda p: p.name)
def test_real_image_approach_a(path, cfg, tmp_path):
    img = read_rgb(path)
    seg = build("approach_a", cfg)
    mask, unc, proba = seg.predict_mask(img)
    d = save_annotation_outputs(tmp_path / path.stem, img, mask, unc, proba, image_id=path.stem,
                                approach=seg.describe(), cfg=cfg)
    assert read_mask(d / "multiclass_mask.png").shape == img.shape[:2]
    s = json.loads((d / "stats.json").read_text())
    assert s["total_pixels"] == img.shape[0] * img.shape[1]
    assert np.isfinite(proba).all()
