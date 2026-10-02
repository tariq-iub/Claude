import math
import numpy as np
import pytest
import torch
from vpc.geometry.cylinder import fit_cylinder_from_mask, cylinder_normals, expected_highlight_lateral_position
from vpc.geometry.curvature import normal_discontinuity, mean_curvature_proxy
from vpc.optics.reflection import (diffuse_dolp, specular_stokes_environment, render_stokes, RenderInputs, zenith_azimuth)
from vpc.optics.fresnel import preset
from vpc.optics.microfacet import specular_stokes_cook_torrance, ggx_D
from vpc.optics.stokes import aolp
from vpc.inverse.latent_optics import LatentOpticsHead, PhysicsStokesLayer
from vpc.inverse.specular_diffuse import specular_free_decomposition


def make_mask(h=96, w=96, alpha=math.radians(90), R=20, cx=48, cy=48):
    yy, xx = np.mgrid[:h, :w]
    x, y = xx - cx, -yy - (-cy)
    t = -math.sin(alpha) * x + math.cos(alpha) * y
    return np.abs(t) <= R


@pytest.mark.parametrize("alpha_deg", [90, 60, 20])
def test_cylinder_fit_and_normals(alpha_deg):
    alpha = math.radians(alpha_deg)
    m = make_mask(alpha=alpha)
    # restrict to a finite-length body along the axis
    yy, xx = np.mgrid[:96, :96]
    ax = (xx - 48) * math.cos(alpha) + (-yy + 48) * math.sin(alpha)
    m &= np.abs(ax) < 35
    p = fit_cylinder_from_mask(m)
    d = abs(((p["alpha"] - alpha + math.pi / 2) % math.pi) - math.pi / 2)
    assert d < math.radians(3)
    assert p["radius"] == pytest.approx(20, abs=1.5)
    n, s = cylinder_normals(96, 96, p["alpha"], p["cx"], p["cy_up"], p["radius"])
    assert torch.allclose(n.norm(dim=0), torch.ones(96, 96), atol=1e-5)
    assert (n[2] >= -1e-6).all()
    # centre line normal points at the camera
    assert n[2][m].max().item() > 0.99


def test_vertical_cylinder_normal_x():
    n, s = cylinder_normals(32, 32, math.pi / 2, 16.0, -16.0, 10.0)
    row = n[:, 16, :]
    assert row[0, 16].item() == pytest.approx(0.0, abs=0.1)
    assert row[0, 22].item() == pytest.approx(0.6, abs=0.06)  # s = 6/10
    assert abs(row[1, 22].item()) < 1e-6


def test_highlight_position_symmetry_and_alignment_with_half_vector():
    assert expected_highlight_lateral_position([0, 0, 1], math.pi / 2) == pytest.approx(0.0, abs=1e-9)
    for alpha in (math.pi / 2, math.radians(35)):
        l = np.array([0.5, 0.1, 0.8]); l /= np.linalg.norm(l)
        h = l + np.array([0, 0, 1.0]); h /= np.linalg.norm(h)
        s_star = expected_highlight_lateral_position(l, alpha)
        # pixel at lateral offset s_star on a cylinder centred in a 64x64 image
        R = 20.0
        ta = np.array([-math.sin(alpha), math.cos(alpha)])
        x, y = s_star * R * ta  # image-plane offset from centre (y up)
        n, _ = cylinder_normals(64, 64, alpha, 32.0, -32.0, R)
        col, row = int(round(32 + x)), int(round(32 - y))
        nh = float((n[:, row, col].numpy() * h[:]).sum())
        # the facet at s_star has the largest n.h along the lateral line (h has an axial component the cylinder cannot match)
        h_perp = h - np.dot(h, np.array([math.cos(alpha), math.sin(alpha), 0])) * np.array([math.cos(alpha), math.sin(alpha), 0])
        assert nh == pytest.approx(np.linalg.norm(h_perp), abs=0.03)


def test_diffuse_dolp_properties():
    th = torch.linspace(0, math.pi / 2, 50, dtype=torch.float64)
    d = diffuse_dolp(th, 1.5)
    assert d[0].item() == pytest.approx(0, abs=1e-9)
    assert (d[1:] >= d[:-1] - 1e-12).all() and d[-1] < 0.5


def test_environment_specular_aolp_is_perpendicular_to_incidence_plane():
    n = torch.tensor([math.sin(0.7) * math.cos(0.4), math.sin(0.7) * math.sin(0.4), math.cos(0.7)])
    nn_, k = preset("copper")
    S = specular_stokes_environment(n, nn_, k, torch.tensor(0.05))
    ang = aolp(S[1].unsqueeze(0).T if False else S[1]) if False else 0.5 * torch.atan2(S[1, 2], S[1, 1])
    expect = (0.4 + math.pi / 2)
    d = (ang - expect) % math.pi
    assert min(d, math.pi - d) < 1e-4
    assert S[1, 1:].norm() < S[1, 0]  # partial polarization


def test_roughness_reduces_dolp():
    n = torch.tensor([0.5, 0.0, math.sqrt(0.75)])
    nn_, k = preset("copper")
    lo = specular_stokes_environment(n, nn_, k, torch.tensor(0.05))
    hi = specular_stokes_environment(n, nn_, k, torch.tensor(0.9))
    assert hi[1, 1:].norm() < lo[1, 1:].norm()


def test_cook_torrance_pol_matches_mueller_path_for_unpolarized():
    n = torch.tensor([[0.1, 0.2, 0.97]]); n = n / n.norm(dim=-1, keepdim=True)
    l = torch.tensor([[0.3, 0.2, 0.9]]); l = l / l.norm(dim=-1, keepdim=True)
    v = torch.tensor([[0.0, 0.0, 1.0]])
    eta, kap = preset("copper")
    S_fast, _ = specular_stokes_cook_torrance(n, l, v, torch.tensor([0.3]), eta.view(1, 3), kap.view(1, 3))
    S_in = torch.tensor([[1.0, 0, 0, 0]])
    S_full, _ = specular_stokes_cook_torrance(n, l, v, torch.tensor([0.3]), eta.view(1, 3), kap.view(1, 3), S_in=S_in)
    assert torch.allclose(S_fast, S_full, atol=1e-5)
    ang = 0.5 * torch.atan2(S_fast[0, 1, 2], S_fast[0, 1, 1])
    expect = math.atan2(0.2, 0.3) + math.pi / 2
    d = (ang - expect) % math.pi
    assert min(d, math.pi - d) < 1e-3


def test_ggx_normalization_peak_at_normal():
    a = torch.tensor(0.3)
    assert ggx_D(torch.tensor(1.0), a) > ggx_D(torch.tensor(0.8), a)


def _inputs(h=12, w=12):
    n, _ = cylinder_normals(h, w, math.pi / 2, w / 2, -h / 2, w * 0.45)
    nn_, k = preset("copper")
    return RenderInputs(
        albedo=torch.full((h, w, 3), 0.5), normals=n.permute(1, 2, 0), rough=torch.full((h, w), 0.3),
        eta=nn_.expand(h, w, 3), kappa=k.expand(h, w, 3), spec_weight=torch.ones(h, w), diff_weight=torch.ones(h, w) * 0.2,
        scatter=torch.full((h, w, 3), 0.01), n_diff=torch.full((h, w), 1.8))


@pytest.mark.parametrize("mode", ["environment", "directional"])
def test_render_stokes_physical(mode):
    out = render_stokes(_inputs(), torch.tensor([0.3, 0.2, 0.9]), mode=mode)
    S = out["S"]
    assert S.shape == (12, 12, 3, 3)
    assert torch.allclose(S, out["S_diffuse"] + out["S_specular"] + out["S_scatter"])
    pol = torch.sqrt(S[..., 1] ** 2 + S[..., 2] ** 2)
    assert (pol <= S[..., 0] + 1e-6).all() and (S[..., 0] >= 0).all()


def test_latent_physics_layer_gradients_and_realizability():
    head, phys = LatentOpticsHead(6), PhysicsStokesLayer(residual_gain=0.2)
    f = torch.randn(1, 6, 10, 10, requires_grad=True)
    o = head(f)
    S = phys(o)
    assert (torch.sqrt(S[:, :, 1] ** 2 + S[:, :, 2] ** 2) <= S[:, :, 0] + 1e-5).all()
    # total radiance preserved by sampling
    z = head.sample(o)
    assert torch.allclose(z["D"] + z["S"], o["D"] + o["S"], atol=1e-4)
    S.square().mean().backward()
    assert f.grad.abs().sum() > 0


def test_normal_discontinuity_zero_on_flat_high_at_edge():
    n = torch.zeros(1, 3, 8, 8); n[:, 2] = 1
    assert normal_discontinuity(n).max() < 1e-3
    n[:, 0, :, 4:] = 0.8; n[:, 2, :, 4:] = 0.6
    assert normal_discontinuity(n).max() > 0.5


def test_specular_free_decomposition_nonneg_and_conservative():
    lin = torch.rand(1, 3, 16, 16) * 0.5
    lin[:, :, 4:6, 4:6] += 0.45  # neutral highlight
    r = specular_free_decomposition(lin)
    assert (r["diffuse"] >= 0).all() and (r["specular"] >= 0).all()
    assert torch.allclose(r["diffuse"] + r["specular"], lin, atol=1e-6)
    assert r["spec_scalar"][:, :, 4:6, 4:6].mean() > r["spec_scalar"][:, :, 10:, 10:].mean()
