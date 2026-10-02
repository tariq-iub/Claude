import numpy as np
import pytest
import torch
from vpc.color import (srgb_to_linear, linear_to_srgb, rgb_to_hsv, rgb_to_lab, hsv_features, ciede2000, color_features,
                       color_channels)


def test_srgb_roundtrip():
    x = torch.rand(2, 3, 5, 5)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-4)


def test_lab_white_black_red():
    w = rgb_to_lab(torch.ones(1, 3, 1, 1))[0, :, 0, 0]
    assert w[0].item() == pytest.approx(100.0, abs=0.1) and abs(w[1].item()) < 0.1 and abs(w[2].item()) < 0.1
    assert rgb_to_lab(torch.zeros(1, 3, 1, 1))[0, 0, 0, 0].item() == pytest.approx(0, abs=0.1)
    r = rgb_to_lab(torch.tensor([1.0, 0, 0]).view(1, 3, 1, 1))[0, :, 0, 0]
    assert r[0].item() == pytest.approx(53.24, abs=0.3) and r[1].item() == pytest.approx(80.09, abs=0.5)


def test_hsv_known():
    px = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0.5]]).T.reshape(1, 3, 4, 1)
    h = rgb_to_hsv(px)[0, :, :, 0]
    assert h[0, 0].item() == pytest.approx(0, abs=1e-5) and h[0, 1].item() == pytest.approx(1 / 3, abs=1e-5)
    assert h[0, 2].item() == pytest.approx(2 / 3, abs=1e-5) and h[1, 3].item() == pytest.approx(0, abs=1e-5)
    f = hsv_features(px)
    assert f.shape[1] == 4 and torch.allclose(f[0, 0] ** 2 + f[0, 1] ** 2, torch.ones(4, 1))


def test_ciede2000_sharma_reference_pairs():
    # Sharma, Wu, Dalal (2005) test data pairs 1 and 2
    a = np.array([[50.0, 2.6772, -79.7751], [50.0, 3.1571, -77.2803]])
    b = np.array([[50.0, 0.0, -82.7485], [50.0, 0.0, -82.7485]])
    d = ciede2000(a, b)
    assert d[0] == pytest.approx(2.0425, abs=1e-3) and d[1] == pytest.approx(2.8615, abs=1e-3)
    assert ciede2000(a[:1], a[:1])[0] == pytest.approx(0, abs=1e-9)


def test_color_feature_selection():
    x = torch.rand(1, 3, 4, 4)
    assert color_features(x, ("rgb", "lab", "hsv")).shape[1] == color_channels(("rgb", "lab", "hsv")) == 10
    with pytest.raises(ValueError):
        color_features(x, ("xyz",))
