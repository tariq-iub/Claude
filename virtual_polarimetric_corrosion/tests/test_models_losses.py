import math
import numpy as np
import pytest
import torch
from vpc.models.vp_corrosion_net import VPCorrosionNet, NetConfig
from vpc.models.teacher import PolarimetricTeacher, measured_stokes
from vpc.models.student import StudentVPCorrosionNet
from vpc.models.baseline import UNetLite, rgb_only_mobile, build_baseline, pseudo_polarizer, apply_clahe, apply_gamma, retinex_msr, \
    highlight_suppression, lab_threshold_segmentation, hsv_threshold_segmentation, physics_only_virtual_polarizer
from vpc.losses import (circular_polarization_loss, periodic_loss, physics_constraint_loss, reconstruction_loss, distillation_loss,
                        total_loss, uncertainty_loss)
from vpc.polarization.virtual_analyzer import VirtualAnalyzer
from vpc.polarization.uncertainty import mc_dropout_predict
from vpc.corrosion.segmentation import dice_loss, seg_loss, boundary_loss
from vpc.corrosion.severity import area_to_level, ordinal_loss, decode_ordinal
from vpc.corrosion.pitting import pit_cue_maps
from vpc.experiments.synthetic import SynthConfig, make_group, render_view

X = torch.rand(2, 3, 64, 64)


@pytest.mark.parametrize("fusion", ["early", "mid", "late"])
def test_forward_backward_all_fusions(fusion):
    m = VPCorrosionNet(NetConfig(fusion=fusion, cartridge_prior=True))
    n = torch.nn.functional.normalize(torch.rand(2, 3, 64, 64), dim=1)
    o = m(X, n)
    for k in ("seg_logits", "pit_logit", "seg_logvar", "S_hat", "stack", "severity", "severity_ordinal", "lin_rec"):
        assert k in o and torch.isfinite(o[k]).all(), k
    assert o["seg_logits"].shape == (2, 7, 64, 64) and o["S_hat"].shape == (2, 3, 3, 64, 64)
    (o["seg_logits"].mean() + o["S_hat"].mean() + o["severity"].mean()).backward()
    assert all(p.grad is not None for p in m.optics.head.raw.parameters())


@pytest.mark.parametrize("spaces", [("rgb",), ("lab",), ("hsv",), ("rgb", "lab", "hsv")])
def test_colour_space_options(spaces):
    m = VPCorrosionNet(NetConfig(color_spaces=spaces))
    assert m(X)["seg_logits"].shape[1] == 7


def test_rgb_only_has_no_optical_branch_and_fewer_params():
    r, f = rgb_only_mobile(), VPCorrosionNet(NetConfig())
    assert r.optics is None and sum(p.numel() for p in r.parameters()) < sum(p.numel() for p in f.parameters())
    assert "S_hat" not in r(X)


def test_stokes_of_model_physical_and_periodic_analyzer():
    m = VPCorrosionNet().eval()
    o = m(X)
    S = o["S_hat"]
    assert (torch.sqrt(S[:, :, 1] ** 2 + S[:, :, 2] ** 2) <= S[:, :, 0] + 1e-5).all()
    va = VirtualAnalyzer()
    th = torch.tensor([10.0, 77.0, 150.0])
    assert periodic_loss(lambda t: va(S, t), th, total=S[:, :, 0].unsqueeze(1)).item() < 1e-9


def test_periodic_loss_detects_nonperiodic_response():
    bad = lambda t: torch.cos(t * math.pi / 180).view(-1, 1)      # period 360 not 180
    assert periodic_loss(bad, torch.tensor([10.0, 80.0])).item() > 0.1


def test_circular_loss_period_pi_invariance():
    S = torch.rand(1, 1, 3, 4, 4); S[:, :, 0] += 2
    ang = 0.7
    base = torch.stack([S[:, :, 0], 0.5 * S[:, :, 0] * math.cos(2 * ang), 0.5 * S[:, :, 0] * math.sin(2 * ang)], 2)
    shifted = torch.stack([S[:, :, 0], 0.5 * S[:, :, 0] * math.cos(2 * (ang + math.pi)), 0.5 * S[:, :, 0] * math.sin(2 * (ang + math.pi))], 2)
    assert circular_polarization_loss(base, shifted)["aolp"].item() < 1e-5
    ortho = torch.stack([S[:, :, 0], -base[:, :, 1], -base[:, :, 2]], 2)   # AoLP + 90 deg
    assert circular_polarization_loss(base, ortho)["aolp"].item() > 1.5      # cos(2*pi/2) = -1 -> loss ~2


def test_total_loss_runs_with_synthetic_batch_and_teacher():
    cfg = SynthConfig(size=64)
    d = [render_view(make_group(i, cfg), 10 + i, cfg) for i in range(2)]
    from vpc.experiments.datasets import collate
    b = collate([{**x, "group_id": 0} for x in d])
    m = VPCorrosionNet()
    out = m(b["rgb"]); out["lin"] = b["lin"]
    T = PolarimetricTeacher(angles_deg=b["angles"][0].tolist())
    to = T(b["stack"])
    loss, parts = total_loss(out, b, teacher_out=to)
    assert torch.isfinite(loss) and {"seg", "pit", "cls", "edge", "unc", "rec", "phys", "pol", "distill"} <= set(parts)
    loss.backward()


def test_teacher_measured_stokes_matches_ground_truth_on_noise_free_stack():
    cfg = SynthConfig(size=64, read_noise=0.0, shot_noise=0.0, exposure_range=(0.2, 0.3))   # no clipping
    d = render_view(make_group(3, cfg), 4, cfg)
    S = measured_stokes(d["stack"].unsqueeze(0), d["angles"].tolist())[0]
    assert torch.allclose(S, d["S_gt"], atol=2e-3)


def test_distillation_loss_zero_when_equal():
    S = torch.rand(1, 3, 3, 8, 8); S[:, :, 0] += 1.5
    s = {"S_hat": S, "feat_proj": torch.rand(1, 4, 4, 4), "seg_logits": torch.randn(1, 7, 8, 8)}
    t = {"S_meas": S.clone(), "feat": s["feat_proj"].clone(), "seg_logits": s["seg_logits"].clone()}
    assert distillation_loss(s, t)["total"].item() < 1e-6


def test_student_interface():
    m = StudentVPCorrosionNet()
    d = m.distill_outputs(m(X))
    assert set(d) == {"S", "feat", "seg_logits"}


def test_mc_dropout_positive_mutual_information():
    m = VPCorrosionNet(NetConfig(dropout=0.3))
    r = mc_dropout_predict(m, X[:1], n_samples=4)
    assert r["prob"].shape == (1, 7, 64, 64) and (r["mutual_info"] >= 0).all() and r["mutual_info"].max() > 0


def test_seg_losses_and_severity():
    lg = torch.randn(2, 7, 16, 16, requires_grad=True)
    y = torch.randint(0, 7, (2, 16, 16))
    (seg_loss(lg, y) + boundary_loss(lg, y)).backward()
    assert dice_loss(torch.nn.functional.one_hot(y, 7).permute(0, 3, 1, 2).float() * 50, y).item() < 0.01
    lv = area_to_level(torch.tensor([0.0, 0.02, 0.1, 0.2, 0.5]))
    assert lv.tolist() == [0, 1, 2, 3, 4]
    lo = torch.zeros(5, 4); 
    assert ordinal_loss(lo, lv).item() == pytest.approx(math.log(2), abs=1e-5)
    assert decode_ordinal(torch.tensor([[5.0, 5, -5, -5]])).item() == 2


def test_pit_cues_fire_on_synthetic_pit():
    lum = torch.full((1, 1, 32, 32), 0.6); lum[:, :, 15:17, 15:17] = 0.05
    c = pit_cue_maps(lum)
    assert c.shape == (1, 4, 32, 32) and c[0, 0, 14:18, 14:18].mean() > c[0, 0, :8, :8].mean()


def test_baselines_run():
    assert UNetLite()(X)["seg_logits"].shape == (2, 7, 64, 64)
    assert build_baseline("mobile_rgb")(X)["seg_logits"].shape[1] == 7
    with pytest.raises(NotImplementedError):
        build_baseline("yolo")
    img = np.random.rand(48, 48, 3).astype(np.float32)
    for f in (apply_clahe, apply_gamma, retinex_msr, highlight_suppression, pseudo_polarizer):
        o = f(img); assert o.shape == img.shape and np.isfinite(o).all() and o.min() >= -1e-6 and o.max() <= 1 + 1e-6
    assert lab_threshold_segmentation(img).shape == (48, 48) and hsv_threshold_segmentation(img).shape == (48, 48)
    n = torch.zeros(1, 3, 48, 48); n[:, 2] = 1
    r = physics_only_virtual_polarizer(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0), n, 45.0)
    assert r["I_theta"].shape == (1, 3, 48, 48) and torch.isfinite(r["I_theta"]).all()


def test_pseudo_polarizer_is_not_polarization_physics():
    # strawman property: its 'analyzer' response is one global gain, identical for every pixel and unrelated to the scene's
    # polarization, so it cannot express per-pixel DoLP/AoLP (two pixels with different polarization respond identically).
    img = np.full((8, 8, 3), 0.3, np.float32)
    img[:, 4:] = 0.6
    ratios = []
    for a in (0, 30, 45, 90, 135):
        o = pseudo_polarizer(img, a)
        ratios.append(o[:, 4:].mean() / img[:, 4:].mean() - o[:, :4].mean() / img[:, :4].mean())
    assert max(abs(r) for r in ratios) < 1e-6
    assert pseudo_polarizer(img, 10).mean() == pytest.approx(pseudo_polarizer(img, 190).mean())   # period 180 is respected
