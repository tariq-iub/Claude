"""Plotting-code smoke tests. The data here are throw-away TEST FIXTURES written to pytest tmp dirs; they never touch results/ or tables/."""
import os
import numpy as np
import pandas as pd
import pytest
import torch
from vpc.experiments.tables import SCHEMAS, write_templates, append_rows
from vpc.experiments.export import export_predictions
from vpc.experiments.datasets import SyntheticCorrosionDataset
from vpc.experiments.synthetic import SynthConfig
from vpc.experiments.train import TrainConfig
from vpc.models.vp_corrosion_net import VPCorrosionNet, NetConfig
from vpc.models.baseline import UNetLite
from vpc.visualization import figures as F
from vpc.visualization.curves import fit_curve, export_raw_and_fitted, loess
from vpc.visualization import fig01_architecture


def test_schema_templates_are_header_only(tmp_path):
    paths = write_templates(str(tmp_path))
    assert len(paths) == len(SCHEMAS) and all(os.path.exists(p) for p in paths)
    for p in paths:
        df = pd.read_csv(p)
        assert len(df) == 0 and list(df.columns) == SCHEMAS[os.path.basename(p)[:-4]]
    required = {"dataset_summary", "overall_segmentation", "per_class_performance", "pitting_performance", "severity_performance",
                "polarization_reconstruction", "physical_vs_virtual", "ablation", "robustness", "runtime", "memory", "uncertainty", "statistical_tests"}
    assert required <= set(SCHEMAS)


def test_append_rows_rejects_unknown_columns(tmp_path):
    append_rows("ablation", [{"arm": "a", "seed": 0, "miou": 0.5}], str(tmp_path))
    assert pd.read_csv(tmp_path / "ablation.csv").shape[0] == 1
    with pytest.raises(ValueError):
        append_rows("ablation", [{"arm": "a", "bogus": 1}], str(tmp_path))


def test_curve_fitting_methods_and_raw_fitted_export(tmp_path):
    x = np.linspace(0, 10, 25); y = np.sin(x) + 0.05 * np.random.default_rng(0).standard_normal(25)
    for m in ("pchip", "spline", "savgol", "loess", "poly"):
        xf, yf = fit_curve(x, y, m)
        assert np.isfinite(yf).all() and len(xf) == len(yf)
    xp = np.arange(0, 180, 15.0); yp = np.cos(np.radians(2 * xp))
    xf, yf = fit_curve(xp, yp, "periodic_spline", periodic_period=180.0)
    assert abs(yf[0] - np.cos(0)) < 1e-6
    export_raw_and_fitted("t", xp, yp, "pchip", str(tmp_path), series="s")
    assert (tmp_path / "t_raw.csv").exists() and (tmp_path / "t_fitted.csv").exists()
    assert "method" in pd.read_csv(tmp_path / "t_fitted.csv").columns
    # PCHIP must not overshoot monotone data
    xs, ys = np.arange(6.0), np.array([0, 0, 0, 1, 1, 1.0])
    assert fit_curve(xs, ys, "pchip")[1].max() <= 1 + 1e-9


def test_loess_recovers_line():
    x = np.linspace(0, 1, 30)
    assert np.allclose(loess(x, 2 * x + 1, 0.6), 2 * x + 1, atol=1e-6)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 9])
def test_physics_figures_render(tmp_path, n):
    out = F.REGISTRY[n](outdir=str(tmp_path), tables=str(tmp_path))
    assert out and all(os.path.exists(p) for p in out)
    from PIL import Image
    png = [p for p in out if p.endswith(".png")][0]
    assert abs(Image.open(png).info["dpi"][0] - 300) < 1


def test_fig1_exports_three_formats(tmp_path):
    out = fig01_architecture.make(str(tmp_path))
    assert sorted(os.path.splitext(p)[1] for p in out) == [".pdf", ".png", ".svg"]
    from PIL import Image
    assert Image.open([p for p in out if p.endswith(".png")][0]).info["dpi"][0] >= 299.9


def test_result_figures_skip_without_data(tmp_path, capsys):
    for n in (7, 8, 10, 11, 12, 13, 14, 15):
        assert F.REGISTRY[n](outdir=str(tmp_path), results=str(tmp_path / "none")) is None
    assert "SKIPPED" in capsys.readouterr().out


def test_result_figures_render_from_fixture_data(tmp_path):
    res = tmp_path / "res"; res.mkdir()
    rng = np.random.default_rng(0)
    ds = SyntheticCorrosionDataset(range(3), 1, SynthConfig(size=64), seed=0)
    models = {"unet": UNetLite(), "vp": VPCorrosionNet(NetConfig())}
    export_predictions(models, ds, str(res / "predictions"), n=3, cfg=TrainConfig(device="cpu"), data_origin="TEST_FIXTURE")
    pd.DataFrame({"arm": np.repeat(["rgb_only", "+lab"], 3), "seed": [0, 1, 2] * 2, "miou": rng.random(6), "data": "TEST_FIXTURE"}).to_csv(res / "ablation.csv", index=False)
    rows = [{"method": m, "seed": s, "perturbation": p, "level": l, "miou": rng.random(), "mf1": rng.random(), "data_origin": "TEST_FIXTURE"}
            for m in ("a", "b") for s in (0, 1) for p in ("noise_sigma", "glare") for l in (0, 1, 2)]
    pd.DataFrame(rows).to_csv(res / "robustness.csv", index=False)
    pd.DataFrame({"config": np.repeat(["A", "B", "C"], 2), "seed": [0, 1] * 3, "miou": rng.random(6), "pol_dolp_mae": rng.random(6),
                  "latency_ms_cpu": rng.random(6), "data_origin": "TEST_FIXTURE"}).to_csv(res / "physical_vs_virtual.csv", index=False)
    pd.DataFrame({"method": ["a", "b"], "seed": [0, 0], "miou": [0.5, 0.6]}).to_csv(res / "overall_segmentation.csv", index=False)
    pd.DataFrame({"model": ["a", "b"], "device": ["cpu", "cpu"], "mean_ms": [10.0, 20.0]}).to_csv(res / "runtime.csv", index=False)
    pd.DataFrame({"method": "a", "bin_center": np.linspace(0.05, 0.95, 10), "accuracy": rng.random(10), "confidence": np.linspace(0.05, 0.95, 10), "count": 5}).to_csv(res / "reliability.csv", index=False)
    pd.DataFrame({"method": "a", "coverage": np.linspace(0.1, 1, 10), "risk": np.linspace(0.05, 0.3, 10)}).to_csv(res / "risk_coverage.csv", index=False)
    out = tmp_path / "figs"
    for n in (7, 8, 10, 11, 12, 13, 14, 15):
        r = F.REGISTRY[n](outdir=str(out), results=str(res))
        assert r and os.path.exists(r[0]), n
