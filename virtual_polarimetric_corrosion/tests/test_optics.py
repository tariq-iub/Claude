import math
import torch
import pytest
from vpc.optics.fresnel import (dielectric_fresnel, conductor_fresnel, fresnel_reflectance, brewster_angle,
                                fresnel_dolp, principal_angle_of_incidence, preset)
from vpc.optics.stokes import (stokes_from_four, fit_linear_stokes, dolp, aolp, wrapped_angular_diff, is_physical,
                               stokes_from_dolp_aolp, aolp_features)
from vpc.optics.mueller import (jones_to_mueller, linear_polarizer_mueller, rotation_mueller, rotate_mueller,
                                fresnel_mueller, apply_mueller, retarder_mueller, analyzer_intensity_from_mueller)

D = torch.float64


def test_dielectric_normal_incidence():
    Rs, Rp = dielectric_fresnel(torch.tensor(0.0, dtype=D), 1.0, 1.5)
    assert Rs.item() == pytest.approx(0.04, abs=1e-9) and Rp.item() == pytest.approx(0.04, abs=1e-9)


def test_brewster_zero_rp():
    th = torch.tensor(brewster_angle(1.5), dtype=D)
    _, Rp = dielectric_fresnel(th, 1.0, 1.5)
    assert Rp.item() < 1e-12


def test_dielectric_energy_and_grazing():
    th = torch.linspace(0, math.pi / 2 - 1e-6, 200, dtype=D)
    Rs, Rp = dielectric_fresnel(th, 1.0, 1.5)
    assert (Rs <= 1 + 1e-9).all() and (Rp <= 1 + 1e-9).all() and Rs[-1] > 0.99
    assert (Rs >= Rp - 1e-12).all()  # s always reflects at least as much as p for dielectrics


def test_total_internal_reflection():
    th = torch.tensor(math.radians(60), dtype=D)
    Rs, Rp = dielectric_fresnel(th, 1.5, 1.0)  # glass->air beyond critical angle (41.8 deg)
    assert Rs.item() == pytest.approx(1.0, abs=1e-9) and Rp.item() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name_idx", [0, 1, 2])
def test_copper_channels(name_idx):
    n, k = preset("copper", D)
    th = torch.linspace(0, math.pi / 2 - 1e-6, 300, dtype=D)
    Rs, Rp, delta = conductor_fresnel(th, n[name_idx], k[name_idx])
    assert (Rs <= 1 + 1e-9).all() and (Rp <= 1 + 1e-9).all()
    assert Rs[0].item() == pytest.approx(Rp[0].item(), abs=1e-9)  # equal at normal incidence
    assert abs(delta[0].item()) < 1e-9                            # convention: zero retardance at normal incidence
    assert (Rs >= Rp - 1e-9).all()                                # metals: s-dominant
    assert Rs[0].item() > 0.5


def test_copper_red_more_reflective_than_blue_at_normal():
    n, k = preset("copper", D)
    Rs, _, _ = conductor_fresnel(torch.tensor(0.0, dtype=D), n, k)
    assert Rs[0] > Rs[1] > Rs[2]  # reddish tint


def test_principal_angle_between_0_and_90():
    a = principal_angle_of_incidence(0.924, 2.452)
    assert 0.3 < a < 1.5


def test_fresnel_dolp_unpolarized_input_normal_is_zero():
    assert abs(fresnel_dolp(torch.tensor(0.0, dtype=D), 1.5, 0.0).item()) < 1e-9


def test_stokes_four_roundtrip_and_ls():
    S = torch.tensor([1.0, 0.3, -0.2], dtype=D).view(3, 1, 1).expand(3, 4, 4)
    th = torch.tensor([0, 45, 90, 135], dtype=D) * math.pi / 180
    I = 0.5 * (S[0] + S[1] * torch.cos(2 * th).view(-1, 1, 1) + S[2] * torch.sin(2 * th).view(-1, 1, 1))
    S4 = stokes_from_four(*I.unbind(0))
    assert torch.allclose(S4, S)
    th8 = torch.arange(8, dtype=D) * math.pi / 8
    I8 = 0.5 * (S[0] + S[1] * torch.cos(2 * th8).view(-1, 1, 1) + S[2] * torch.sin(2 * th8).view(-1, 1, 1))
    assert torch.allclose(fit_linear_stokes(I8, th8, dim=0), S)


def test_dolp_aolp_values():
    S = stokes_from_dolp_aolp(torch.tensor(2.0, dtype=D), torch.tensor(0.5, dtype=D), torch.tensor(0.3, dtype=D))
    assert dolp(S, eps=1e-12).item() == pytest.approx(0.5, abs=1e-9)
    assert aolp(S).item() == pytest.approx(0.3, abs=1e-9)
    assert is_physical(S)


def test_wrapped_angular_diff_period():
    a = torch.tensor([0.0, 0.1, math.pi / 2 - 0.05, -math.pi / 2 + 0.05], dtype=D)
    b = torch.tensor([math.pi, 0.1 + math.pi, -math.pi / 2 + 0.05, math.pi / 2 - 0.05], dtype=D)
    d = wrapped_angular_diff(a, b)
    assert d[0].item() == pytest.approx(0, abs=1e-9) and d[1].item() == pytest.approx(0, abs=1e-9)
    assert d[2].item() == pytest.approx(0.1, abs=1e-6) and d[3].item() == pytest.approx(0.1, abs=1e-6)
    assert (d <= math.pi / 2 + 1e-9).all()


def test_aolp_features_zero_when_unpolarized():
    S = torch.tensor([1.0, 0.0, 0.0], dtype=D)
    assert torch.allclose(aolp_features(S), torch.zeros(2, dtype=D), atol=1e-6)


def test_jones_to_mueller_horizontal_polarizer():
    J = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
    assert torch.allclose(jones_to_mueller(J), linear_polarizer_mueller(torch.tensor(0.0, dtype=D)).to(D), atol=1e-12)


@pytest.mark.parametrize("deg", [0, 17, 45, 90, 133])
def test_polarizer_rotation_consistency(deg):
    th = torch.tensor(math.radians(deg), dtype=D)
    M0 = linear_polarizer_mueller(torch.tensor(0.0, dtype=D))
    assert torch.allclose(rotate_mueller(M0, th), linear_polarizer_mueller(th), atol=1e-12)


def test_ideal_polarizer_idempotent_and_crossed_extinction():
    M = linear_polarizer_mueller(torch.tensor(0.4, dtype=D))
    assert torch.allclose(M @ M, M, atol=1e-12)
    Mx = linear_polarizer_mueller(torch.tensor(0.4 + math.pi / 2, dtype=D))
    assert torch.allclose(Mx @ M, torch.zeros(4, 4, dtype=D), atol=1e-12)


def test_malus_law():
    S = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=D)  # fully polarized along 0 deg
    for deg in (0, 30, 60, 90):
        th = torch.tensor(math.radians(deg), dtype=D)
        I = analyzer_intensity_from_mueller(S, th)
        assert I.item() == pytest.approx(math.cos(math.radians(deg)) ** 2, abs=1e-12)


def test_finite_extinction_ratio():
    S = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=D)
    I = analyzer_intensity_from_mueller(S, torch.tensor(math.pi / 2, dtype=D), 1.0, 1e-3)
    assert I.item() == pytest.approx(1e-3, abs=1e-9)


def test_fresnel_mueller_unpolarized():
    n, k = preset("copper", D)
    Rs, Rp, de = conductor_fresnel(torch.tensor(0.8, dtype=D), n[1], k[1])
    M = fresnel_mueller(Rs.to(torch.float32), Rp.to(torch.float32), de.to(torch.float32))
    out = apply_mueller(M, torch.tensor([1.0, 0, 0, 0]))
    assert out[0].item() == pytest.approx(0.5 * (Rs + Rp).item(), abs=1e-5)
    assert out[1].item() == pytest.approx(0.5 * (Rs - Rp).item(), abs=1e-5)
    assert abs(out[2].item()) < 1e-6


def test_reflection_mueller_physical():
    # output Stokes of a passive reflector must satisfy S0 >= sqrt(S1^2+S2^2+S3^2)
    n, k = preset("copper", torch.float32)
    Rs, Rp, de = conductor_fresnel(torch.tensor(1.0), n[0], k[0])
    M = fresnel_mueller(Rs, Rp, de)
    for vec in ([0.3, 0.2, 0.1], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]):
        v = torch.tensor(vec, dtype=torch.float32)
        v = v / max(1.0, float(v.norm()))            # |S_pol| <= S0 = 1
        Sin = torch.cat([torch.ones(1), v])
        Sout = apply_mueller(M, Sin)
        assert Sout[0] + 1e-6 >= torch.sqrt((Sout[1:] ** 2).sum())
