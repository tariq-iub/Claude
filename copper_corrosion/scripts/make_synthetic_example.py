"""Generate the synthetic example dataset and a full benchmark run (CPU, ~1 min).
Results are for pipeline verification only - they are NOT corrosion results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cucorr.cli import main  # noqa: E402

out = ROOT / "outputs" / "synthetic_example"
main(["synth", "--out", str(out / "data"), "--n", "40"])
main(["benchmark", "--images", str(out / "data" / "images"), "--masks", str(out / "data" / "masks"),
      "--splits", str(out / "data" / "groups.csv"), "--out", str(out / "benchmark"),
      "--approaches", "A_rgb_euclid,A_lab_de2000,B_rf,B_hgb,C_slic_400,F_hybrid",
      "--experiments", str(ROOT / "configs" / "experiments.yaml"), "--reference-type", "synthetic"])
