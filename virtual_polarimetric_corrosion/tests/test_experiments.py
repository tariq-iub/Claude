import math
import numpy as np
import pytest
import torch
from vpc.experiments.synthetic import SynthConfig, make_group, render_view, generate_dataset
from vpc.experiments.datasets import SyntheticCorrosionDataset, collate
from vpc.experiments.metrics import *
from vpc.experiments.stats import *
from vpc.experiments.splits import group_stratified_split, assert_no_group_leakage, near_duplicate_check
from vpc.experiments.benchmark import count_params, count_macs, benchmark_latency, model_size_mb
from vpc.experiments.robustness import PHOTOMETRIC, rotate, camera_shift, degradation_curve
from vpc.experiments.hardware_io import measured_stokes_np
from vpc.experiments.train import select_device, TrainConfig, train, evaluate
from vpc.models.vp_corrosion_net import VPCorrosionNet, NetConfig
from vpc.experiments.ablation import cumulative_configs, leave_one_out_configs, color_space_configs, fusion_configs


def test_synthetic_ground_truth_consistency():
    cfg = SynthConfig(size=64)
    g = make_group(5, cfg)
    for mode in ("environment", "directional"):
        d = render_view(g, 9, cfg, force_mode=mode)
        S = d["S_gt"]
        assert torch.isfinite(S).all() and (S[:, 0] >= 0).all()
        assert (torch.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2) <= S[:, 0] + 1e-4).all()
        # analyzer stack equals Malus-form of S_gt before sensor noise/clipping: check on unclipped, low-noise render
    cfg2 = SynthConfig(size=64, read_noise=0.0, shot_noise=0.0, exposure_range=(0.1, 0.2), sensor_clip=False)
    d = render_view(make_group(5, cfg2), 9, cfg2)
    th = d["angles"] * math.pi / 180
    c, sn = torch.cos(2 * th).view(-1, 1, 1, 1), torch.sin(2 * th).view(-1, 1, 1, 1)
    S = d["S_gt"]                                            # (3 colour, 3 Stokes, H, W)
    pred = 0.5 * (S[:, 0].unsqueeze(0) + S[:, 1].unsqueeze(0) * c + S[:, 2].unsqueeze(0) * sn)   # (A,3,H,W)
    assert torch.allclose(pred, d["stack"], atol=1e-5)       # analyzer images obey the Malus-form of the ground-truth Stokes field


def test_synthetic_groups_share_texture_across_views_but_not_across_groups():
    cfg = SynthConfig(size=64)
    g1, g2 = make_group(1, cfg), make_group(2, cfg)
    assert not np.array_equal(g1["label"], g2["label"])
    a, b = render_view(g1, 11, cfg), render_view(g1, 12, cfg)
    assert a["meta"]["group_seed"] == b["meta"]["group_seed"] and not torch.equal(a["rgb"], b["rgb"])


def test_severity_range_and_classes():
    cfg = SynthConfig(size=64)
    lo, hi = render_view(make_group(1, cfg, level=0.0), 3, cfg), render_view(make_group(1, cfg, level=1.0), 3, cfg)
    assert lo["severity"] < hi["severity"] <= 1.0


def test_dataset_deterministic():
    ds1, ds2 = SyntheticCorrosionDataset([0, 1], 1, SynthConfig(size=64), seed=7), SyntheticCorrosionDataset([0, 1], 1, SynthConfig(size=64), seed=7)
    assert torch.equal(ds1[1]["rgb"], ds2[1]["rgb"])
    b = collate([ds1[0], ds1[1]])
    assert b["rgb"].shape == (2, 3, 64, 64) and b["stack"].shape == (2, 8, 3, 64, 64)


def test_seg_metrics_known_values():
    m = SegMetrics(2)
    m.update(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    pc = m.per_class()
    assert pc["iou"][1] == pytest.approx(0.5) and pc["recall"][1] == pytest.approx(1.0) and pc["precision"][1] == pytest.approx(0.5)
    assert pc["specificity"][1] == pytest.approx(2 / 3, abs=1e-6) and pc["dice"][1] == pytest.approx(2 / 3, abs=1e-6)
    assert pc["balanced_accuracy"][1] == pytest.approx(0.5 * (1 + 2 / 3), abs=1e-6)


def test_absent_class_excluded_from_means():
    m = SegMetrics(3)
    m.update(np.array([0, 1]), np.array([0, 1]))
    assert m.summary()["miou"] == pytest.approx(1.0)


def test_detection_ap():
    gts = {0: np.array([[0, 0, 10, 10.0]]), 1: np.array([[0, 0, 10, 10.0]])}
    perfect = [(0, 0.9, gts[0][0]), (1, 0.8, gts[1][0])]
    assert average_precision(perfect, gts, 0.5) == pytest.approx(1.0)
    wrong = [(0, 0.9, np.array([50, 50, 60, 60.0])), (1, 0.8, gts[1][0])]
    assert average_precision(wrong, gts, 0.5) == pytest.approx(0.25, abs=1e-6)
    s = map_scores({0: perfect}, {0: gts})
    assert s["mAP50"] == pytest.approx(1.0) and s["mAP50-95"] == pytest.approx(1.0)
    m = np.zeros((20, 20), bool); m[2:6, 3:9] = True
    assert boxes_from_mask(m).tolist() == [[3, 2, 9, 6]]


def test_severity_metrics():
    r = severity_regression([0, 1, 2], [0, 1, 2]); assert r["mae"] == 0 and r["r2"] == pytest.approx(1)
    o = severity_ordinal([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 5); assert o["qwk"] == pytest.approx(1) and o["accuracy"] == 1
    assert quadratic_weighted_kappa([0, 4], [4, 0], 5) < 0


def test_aolp_error_is_circular():
    a = np.array([0.0]); b = np.array([math.pi - 0.05])
    assert aolp_error(a, b) == pytest.approx(0.05, abs=1e-9)
    assert aolp_error(np.array([math.pi / 2 - 0.01]), np.array([-math.pi / 2 + 0.01])) == pytest.approx(0.02, abs=1e-9)


def test_polarimetric_agreement_perfect_and_rotated():
    rng = np.random.default_rng(0)
    S = np.stack([np.ones((6, 6)), 0.3 * np.ones((6, 6)), 0.2 * np.ones((6, 6))])
    r = polarimetric_agreement(S, S)
    assert r["dolp_mae"] < 1e-12 and r["aolp_wrapped_err_deg"] < 1e-9 and r["aolp_cos2_agreement"] == pytest.approx(1)
    Sr = np.stack([S[0], -S[1], -S[2]])                      # AoLP + 90 deg
    assert polarimetric_agreement(Sr, S)["aolp_wrapped_err_deg"] == pytest.approx(90.0, abs=1e-6)


def test_quality_metrics():
    ref = np.clip(np.random.default_rng(0).random((32, 32)) * 0.8, 0, 1); ref[:4, :4] = 1.0
    proc = ref.copy(); proc[:4, :4] = 0.5
    assert glare_suppression_ratio(ref, proc) == pytest.approx(1.0) and highlight_area_reduction(ref, proc) == pytest.approx(1.0)
    assert edge_preservation(ref, ref) == pytest.approx(1.0) and texture_preservation(ref, ref) == pytest.approx(1.0)
    assert 0 < information_entropy(ref) <= 8 and local_contrast(ref) > 0 and gradient_preservation(ref, ref) == pytest.approx(1)
    assert math.isinf(signal_to_glare_ratio(proc))


def test_calibration_metrics():
    conf = np.array([0.9] * 10)
    assert expected_calibration_error(conf, np.array([1] * 9 + [0])) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(conf, np.zeros(10)) == pytest.approx(0.9)
    p = np.array([[0.9, 0.1], [0.2, 0.8]]); y = np.array([0, 1])
    assert brier_score(p, y) == pytest.approx(((0.1 ** 2 + 0.1 ** 2) + (0.2 ** 2 + 0.2 ** 2)) / 2)
    assert negative_log_likelihood(p, y) == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2)
    u = np.arange(100.0); e = (u > 80).astype(float)             # errors concentrated at high uncertainty
    rc = risk_coverage(u, e, 20)
    assert rc["risk"][0] == 0 and rc["risk"][-1] == pytest.approx(0.19) and rc["aurc"] < 0.19
    rd = reliability_diagram_data(np.array([0.1, 0.9]), np.array([0, 1]), 10)
    assert rd["count"].sum() == 2


def test_stats():
    x = [0.7, 0.72, 0.69, 0.71, 0.73]
    r = mean_sd_ci(x); assert r["ci_lo"] < r["mean"] < r["ci_hi"]
    b = bootstrap_ci(x, n_boot=500); assert b["ci_lo"] <= b["estimate"] <= b["ci_hi"]
    g = grouped_bootstrap_ci([1, 1, 1, 5, 5, 5], [0, 0, 0, 1, 1, 1], n_boot=300); assert g["n_groups"] == 2 and g["ci_lo"] >= 1
    a, c = np.array([0.8, 0.82, 0.79, 0.85, 0.81, 0.83, 0.84]), np.array([0.7, 0.71, 0.72, 0.74, 0.69, 0.73, 0.70])
    t = paired_tests(a, c); assert t["mean_diff"] > 0 and t["wilcoxon_p"] < 0.05 and t["cohens_dz"] > 1
    assert paired_bootstrap_diff(a, c, n_boot=500)["ci_lo"] > 0
    assert np.isnan(paired_tests([1, 2, 3], [0, 1, 2])["wilcoxon_p"])        # n<6: refuse to report a meaningless exact p
    h = holm_bonferroni([0.01, 0.04, 0.03]); assert h["p_adj"].tolist() == pytest.approx([0.03, 0.06, 0.06]) and h["reject"].tolist() == [True, False, False]
    assert benjamini_hochberg([0.01, 0.04, 0.03])["reject"].all()
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0


def test_group_split_no_leakage_and_proportions():
    ds = SyntheticCorrosionDataset(range(40), 3, SynthConfig(size=64), seed=2)
    groups = ds.group_ids(); strata = np.array([g % 4 for g in groups])
    sp = group_stratified_split(groups, strata, seed=1)
    assert sum(len(v) for v in sp.values()) == len(groups)
    assert set(groups[sp["train"]]).isdisjoint(groups[sp["test"]]) and set(groups[sp["val"]]).isdisjoint(groups[sp["test"]])
    assert 0.6 < len(sp["train"]) / len(groups) < 0.8
    with pytest.raises(AssertionError):
        assert_no_group_leakage({"train": [1, 2], "test": [2, 3]})


def test_near_duplicate_flags_copies():
    f = np.random.default_rng(0).standard_normal((10, 8)); f[9] = f[0]
    r = near_duplicate_check(f, {"train": np.arange(0, 5), "test": np.array([9, 6])})
    assert r["n_flagged"] == 1


def test_benchmark_utils():
    m = VPCorrosionNet(NetConfig())
    assert count_params(m) > 0 and model_size_mb(m) > 0
    macs = count_macs(m, (torch.rand(1, 3, 64, 64),))
    assert macs > 1e6
    r = benchmark_latency(m, (1, 3, 64, 64), "cuda", warmup=1, iters=3)      # falls back to CPU when no GPU
    assert r["device"] in ("cpu", "cuda") and r["fps"] > 0
    assert select_device("cuda").type in ("cpu", "cuda")


def test_robustness_perturbations():
    x = torch.rand(2, 3, 32, 32)
    for name, (fn, lv) in PHOTOMETRIC.items():
        y = fn(x, lv[1])
        assert y.shape == x.shape and torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1 + 1e-6, name
    mask = torch.randint(0, 3, (2, 32, 32))
    xr, mr = rotate(x, 30, mask); assert xr.shape == x.shape and mr.shape == mask.shape
    xs, ms = camera_shift(x, 3, 2, mask); assert torch.equal(ms, torch.roll(mask, (2, 3), (1, 2)))
    rows = degradation_curve(lambda z: (z.mean(1) > 0.5).long(), x, mask, "noise_sigma", [0.0, 0.1])
    assert len(rows) == 2 and rows[0]["perturbation"] == "noise_sigma"


def test_hardware_io_stokes_numpy_matches_torch():
    from vpc.models.teacher import measured_stokes
    angles = np.array([0, 45, 90, 135.0])
    th = np.radians(angles)
    S = np.array([1.0, 0.3, -0.1])
    stack = 0.5 * (S[0] + S[1] * np.cos(2 * th) + S[2] * np.sin(2 * th))[:, None, None] * np.ones((4, 3, 3))
    assert np.allclose(measured_stokes_np(stack, angles)[:, 0, 0], S)


def test_ablation_arms_are_cumulative_and_valid():
    arms = cumulative_configs()
    names = list(arms)
    assert names[0] == "rgb_only" and not arms["rgb_only"].needs_optics and arms[names[-1]].cartridge_prior
    assert arms["+psrf"].psrf and arms["+virtual_stokes"].vstokes and not arms["+virtual_stokes"].analyzer
    full = arms[names[-1]]
    assert len(leave_one_out_configs(full)) >= 10 and len(color_space_configs(full)) == 6 and len(fusion_configs(full)) == 3
    for nc in list(arms.values())[:4]:
        assert VPCorrosionNet(nc)(torch.rand(1, 3, 32, 32))["seg_logits"].shape == (1, 7, 32, 32)


def test_train_loop_reduces_loss_smoke():
    tr = SyntheticCorrosionDataset(range(4), 1, SynthConfig(size=64), seed=0)
    m = VPCorrosionNet(NetConfig())
    r = train(m, tr, TrainConfig(epochs=6, batch_size=4, device="cpu", lr=3e-3))
    h = [x["loss"] for x in r["history"]]
    assert h[-1] < h[0]
    assert "miou" in evaluate(m, tr, TrainConfig(device="cpu"))["summary"]


def test_validation_harness_identities_are_exact_when_unclipped():
    from vpc.experiments.pipeline import synthetic_validation
    rows = {r["check"]: r for r in synthetic_validation(n_samples=3, size=64, seed=1)}
    for k in ("stack_vs_malus", "four_angle_stokes", "ls_stokes_8", "periodicity", "orthogonal_sum"):
        assert rows[k]["max_over_samples"] < 1e-4, k
    assert rows["psrf_order1_resid"]["max_over_samples"] < 1e-4 and rows["psrf_order2_ho_energy"]["max_over_samples"] < 1e-4
    assert rows["physics_only_untrained::dolp_mae"]["data_origin"] == "SYNTHETIC"


def test_sensor_clipping_switch():
    cfg = SynthConfig(size=64, sensor_clip=False, exposure_range=(2.0, 2.5))
    d = render_view(make_group(2, cfg), 3, cfg)
    assert d["lin"].max() > 1.0
    cfg2 = SynthConfig(size=64, sensor_clip=True, exposure_range=(2.0, 2.5))
    assert render_view(make_group(2, cfg2), 3, cfg2)["lin"].max() <= 1.0
