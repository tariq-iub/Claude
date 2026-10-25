import numpy as np

from cucorr.color_science.color import delta_e, gray_world, lab_to_rgb, rgb_to_lab


def test_lab_white_and_roundtrip():
    assert np.allclose(rgb_to_lab(np.array([255, 255, 255], np.uint8)), [100, 0, 0], atol=0.05)
    rgb = np.random.default_rng(0).integers(0, 256, (50, 3)).astype(np.uint8)
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1


def test_ciede2000_reference_pair():
    # Sharma, Wu & Dalal (2005) test data, pair 1: dE00 = 2.0425
    d = delta_e(np.array([50.0, 2.6772, -79.7751]), np.array([50.0, 0.0, -82.7485]), "ciede2000")
    assert abs(float(d) - 2.0425) < 1e-3


def test_gray_world_neutralizes_cast():
    img = np.full((10, 10, 3), (200, 100, 100), np.uint8)
    out = gray_world(img).reshape(-1, 3).mean(0)
    assert np.ptp(out) < 2
