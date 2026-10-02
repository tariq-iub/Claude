import math
import numpy as np
import pytest
import torch
from vpc.polarization.virtual_analyzer import VirtualAnalyzer, virtual_stack, CANONICAL_ANGLES_DEG, dense_angles_deg
from vpc.polarization.psrf import PSRF, select_order
from vpc.polarization.angle_search import optimise_analyzer
from vpc.polarization.uncertainty import summarize_samples, heteroscedastic_nll
from vpc.optics.stokes import stokes_from_four

D = torch.float64


def rand_stokes(b=2, c=3, h=8, w=8, dtype=torch.float32):
    s0 = torch.rand(b, c, h, w, dtype=dtype) + 0.5
    rho = torch.rand(b, c, h, w, dtype=dtype) * 0.9
    phi = (torch.rand(b, c, h, w, dtype=dtype) - 0.5) * math.pi
    return torch.stack([s0, s0 * rho * torch.cos(2 * phi), s0 * rho * torch.sin(2 * phi)], 2)


def test_periodicity_180():
    S, va = rand_stokes(), VirtualAnalyzer()
    th = torch.tensor([3.0, 41.5, 100.0, 179.0])
    assert torch.allclose(va(S, th), va(S, th + 180.0), atol=1e-5)
    assert torch.allclose(va(S, th), va(S, th - 360.0), atol=1e-5)


def test_orthogonal_sum_equals_S0():
    S, va = rand_stokes(), VirtualAnalyzer()
    th = torch.tensor([0.0, 30.0, 77.0])
    assert torch.allclose(va(S, th) + va(S, th + 90.0), S[:, :, 0].unsqueeze(1).expand(-1, 3, -1, -1, -1), atol=1e-5)


def test_closed_form_equals_mueller_reference():
    va = VirtualAnalyzer()
    S4 = torch.cat([rand_stokes(1, 1, 4, 4)[:, 0].permute(0, 2, 3, 1), torch.zeros(1, 4, 4, 1)], -1)  # (1,4,4,4)
    th = 37.0
    I_cf = va(S4[..., :3].permute(0, 3, 1, 2).unsqueeze(1), th)[:, 0, 0]
    I_ref = va.reference_mueller(S4, th)
    assert torch.allclose(I_cf, I_ref, atol=1e-5)


def test_mueller_mode_matches_closed_form():
    S = rand_stokes()
    a, b = VirtualAnalyzer("closed_form"), VirtualAnalyzer("mueller", t_min=0.0)
    assert torch.allclose(a(S, CANONICAL_ANGLES_DEG), b(S, CANONICAL_ANGLES_DEG), atol=1e-6)


def test_extinction_ratio_leaks():
    S = torch.tensor([1.0, 1.0, 0.0]).view(1, 1, 3, 1, 1)  # fully polarized at 0 deg
    va = VirtualAnalyzer(t_min=1e-2)
    assert va(S, 90.0).item() == pytest.approx(1e-2, abs=1e-6)


def test_differentiable_gradcheck():
    S = rand_stokes(1, 1, 2, 2, torch.float64).requires_grad_(True)
    va = VirtualAnalyzer()
    th = torch.tensor([10.0, 80.0], dtype=D)
    assert torch.autograd.gradcheck(lambda s: va(s, th), (S,))


def test_four_angle_stack_recovers_stokes():
    S = rand_stokes(1, 3, 4, 4)
    st = virtual_stack(S, (0, 45, 90, 135))
    S_rec = stokes_from_four(st[:, 0], st[:, 1], st[:, 2], st[:, 3])  # (3[S],B,C,H,W)
    assert torch.allclose(S_rec.permute(1, 2, 0, 3, 4), S, atol=1e-5)


def test_dense_angles():
    a = dense_angles_deg(5.0)
    assert len(a) == 36 and a[0] == 0 and a[-1] == 175


def test_psrf_first_order_exact():
    S = rand_stokes(1, 1, 5, 5)
    st = virtual_stack(S, CANONICAL_ANGLES_DEG)[:, :, 0]  # (1,A,H,W)
    m = PSRF(CANONICAL_ANGLES_DEG, 1)
    c = m.fit(st)
    assert torch.allclose(c[:, 0], 0.5 * S[:, 0, 0], atol=1e-5)
    assert torch.allclose(c[:, 1], 0.5 * S[:, 0, 1], atol=1e-5)
    assert torch.allclose(c[:, 2], 0.5 * S[:, 0, 2], atol=1e-5)
    f = m.features(c)
    rho = torch.sqrt(S[:, 0, 1] ** 2 + S[:, 0, 2] ** 2) / S[:, 0, 0]
    assert torch.allclose(f["dolp_hat"], rho, atol=1e-4)


def test_psrf_second_order_zero_for_ideal_physics_and_requires_5_angles():
    S = rand_stokes(1, 1, 3, 3)
    st = virtual_stack(S, CANONICAL_ANGLES_DEG)[:, :, 0]
    c = PSRF(CANONICAL_ANGLES_DEG, 2).fit(st)
    assert c[:, 3:].abs().max() < 1e-5   # ideal linear analyzer has no 4-theta harmonic
    with pytest.raises(ValueError):
        PSRF((0, 45, 90, 135), 2)         # 5 unknowns, 4 angles (and cos4t/sin4t aliasing)


def test_psrf_detects_injected_second_harmonic():
    th = torch.tensor(CANONICAL_ANGLES_DEG) * math.pi / 180
    base = 0.5 + 0.2 * torch.cos(2 * th - 0.3)
    sig = (base + 0.05 * torch.cos(4 * th)).view(1, -1, 1, 1).expand(1, 8, 2, 2).contiguous()
    res = select_order(sig, CANONICAL_ANGLES_DEG, max_order=2, holdout_every=4)
    # with only 8 angles and holdout every 4 the order-2 fit has 6 train angles for 5 params: it should win clearly
    assert res["best"] == 2


def test_angle_search_finds_designed_optimum_and_period():
    A = np.arange(0, 180, 15.0)
    H = W = 24
    yy, xx = np.mgrid[:H, :W]
    mask = xx > W // 2
    stack = []
    for a in A:
        contrast = 0.4 * np.cos(math.radians(2 * (a - 60.0)))   # best contrast at 60 deg
        im = 0.5 + contrast * (mask - 0.5) + 0.01 * np.random.default_rng(0).standard_normal((H, W))
        stack.append(np.clip(im, 0, 1))
    r = optimise_analyzer(np.stack(stack), A, mask=mask, weights=(0, 1, 0, 0, 0))
    assert abs(r.theta_star_deg - 60.0) < 8.0
    assert r.J_fit.shape == r.fine_angles_deg.shape
    assert abs(r.J_fit[0] - r.J_fit[-1]) < 0.2  # smooth across the 180-deg wrap


def test_summarize_and_nll():
    s = torch.randn(200, 4)
    out = summarize_samples(s, 0.9)
    assert (out["lo"] <= out["mean"]).all() and (out["hi"] >= out["mean"]).all()
    nll = heteroscedastic_nll(torch.zeros(3), torch.zeros(3), torch.ones(3))
    assert nll.item() == pytest.approx(0.5)
