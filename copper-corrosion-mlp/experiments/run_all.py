"""Orchestrates the full pipeline in order. Each stage writes its own
results/ artifacts; re-running is idempotent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import feature_ablation, architecture_ablation, ablation_study, optimize_mlp, deployment_check

if __name__ == "__main__":
    print("=== 1/5 feature ablation ===")
    feature_ablation.run()
    print("\n=== 2/5 architecture ablation ===")
    architecture_ablation.run()
    print("\n=== 3/5 staged ablation study (vs Prompt.md baseline) ===")
    ablation_study.run()
    print("\n=== 4/5 final model: multi-seed training + evaluation + calibration + uncertainty + interpretability ===")
    optimize_mlp.run()
    print("\n=== 5/5 deployment check: dynamic INT8 quantization ===")
    deployment_check.run()
