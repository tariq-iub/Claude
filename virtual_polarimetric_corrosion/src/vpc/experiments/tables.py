"""Results-table schemas (CSV headers only). Tables are filled by experiment scripts from real runs; templates contain NO values."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

import pandas as pd

SCHEMAS: Dict[str, List[str]] = {
    "dataset_summary": ["split", "n_groups", "n_images", "n_pol_stack_images", "mean_corroded_fraction", "pitting_present_fraction",
                        "cameras", "illumination_conditions", "annotation_protocol", "data_origin"],
    "overall_segmentation": ["method", "input", "seed", "miou", "mdice", "mprecision", "mrecall", "mspecificity", "mf1",
                             "mbalanced_accuracy", "pixel_accuracy", "data_split", "n_test_images", "data_origin"],
    "per_class_performance": ["method", "seed", "class", "iou", "dice", "precision", "recall", "specificity", "f1", "balanced_accuracy",
                              "support_px", "data_origin"],
    "pitting_performance": ["method", "seed", "precision", "recall", "f1", "iou", "map50", "map50_95", "min_pit_area_px", "data_origin"],
    "severity_performance": ["method", "seed", "mae", "rmse", "r2", "accuracy", "off_by_one_accuracy", "qwk", "spearman", "area_pct_mae",
                             "data_origin"],
    "polarization_reconstruction": ["method", "seed", "sample_id", "angle_deg", "img_rmse", "img_psnr", "img_ssim", "dolp_mae",
                                    "aolp_wrapped_err_deg", "s0_rel_err", "data_origin"],
    "physical_vs_virtual": ["config", "seed", "miou", "mf1", "severity_mae", "pol_dolp_mae", "pol_aolp_wrapped_err_deg", "glare_supp_phys",
                            "glare_supp_virt", "hl_area_red_phys", "hl_area_red_virt", "edge_preservation", "texture_preservation",
                            "latency_ms_cpu", "data_origin"],
    "ablation": ["arm", "seed", "miou", "mdice", "mf1", "mbalanced_accuracy", "severity_mae", "params", "distilled", "data"],
    "robustness": ["method", "seed", "perturbation", "level", "miou", "mf1", "data_origin"],
    "runtime": ["model", "device", "threads", "batch", "input", "mean_ms", "median_ms", "p95_ms", "fps"],
    "memory": ["model", "params", "size_mb", "macs_G", "ram_mb", "vram_mb", "device"],
    "uncertainty": ["method", "seed", "ece", "brier", "nll", "aurc", "spearman_std_vs_abs_error", "mean_entropy_correct", "mean_entropy_wrong",
                    "data_origin"],
    "statistical_tests": ["comparison", "metric", "n_units", "unit", "mean_diff", "ci_lo", "ci_hi", "cohens_dz", "wilcoxon_p", "p_adj_holm",
                          "correction_family"],
    # raw curves behind figures
    "angle_objective": ["sample_id", "angle_deg", "T", "C", "E", "G", "U", "J", "data_origin"],
    "reliability": ["method", "bin_center", "accuracy", "confidence", "count"],
    "risk_coverage": ["method", "coverage", "risk"],
}


def write_templates(outdir: str = "tables", overwrite: bool = False) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    out = []
    for name, cols in SCHEMAS.items():
        p = os.path.join(outdir, f"{name}.csv")
        if overwrite or not os.path.exists(p):
            pd.DataFrame(columns=cols).to_csv(p, index=False)
        out.append(p)
    return out


def append_rows(name: str, rows: Iterable[dict], outdir: str = "results", data_origin: str | None = None) -> str:
    """Append raw rows to results/<name>.csv; unknown columns are rejected so schemas stay authoritative."""
    cols = SCHEMAS[name]
    df = pd.DataFrame(list(rows))
    if data_origin is not None and "data_origin" in cols and "data_origin" not in df:
        df["data_origin"] = data_origin
    unknown = set(df.columns) - set(cols)
    if unknown:
        raise ValueError(f"columns {sorted(unknown)} not in schema '{name}'")
    for c in cols:
        if c not in df:
            df[c] = float("nan")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{name}.csv")
    df[cols].to_csv(p, mode="a", index=False, header=not os.path.exists(p))
    return p
