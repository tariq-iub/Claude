"""Single evaluation pipeline: every approach, same held-out images, same metrics."""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..reporting.outputs import save_annotation_outputs
from ..schema import CLASS_NAMES, NUM_CLASSES, UNCERTAIN
from ..segmentation.base import Segmenter
from .metrics import confusion, coverage_risk, evaluate_segmentation, scores_from_confusion
from .statistics import compare_approaches, mean_ci, select_best

SUMMARY_KEYS = ("macro_miou", "corrosion_miou", "healthy_iou", "macro_dice", "fw_dice", "binary_dice",
                "binary_iou", "sensitivity", "specificity", "boundary_f1", "hd95", "rare_class_iou",
                "small_region_recall", "balanced_accuracy", "abstention_rate", "ece", "brier")


def perturb(image: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Realistic acquisition changes for robustness testing."""
    f = image.astype(np.float32)
    if kind == "brightness":
        f *= 1.25
    elif kind == "dark":
        f *= 0.75
    elif kind == "white_balance":
        f *= np.array([1.08, 1.0, 0.9])
    elif kind == "gamma":
        f = 255 * (f / 255) ** 1.3
    elif kind == "shadow":
        h, w = f.shape[:2]
        f *= (0.6 + 0.4 * (np.arange(w) / max(w - 1, 1)))[None, :, None]
    elif kind == "specular":
        h, w = f.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        f += 120 * np.exp(-(((yy - h / 3) ** 2 + (xx - w / 3) ** 2) / (2 * (0.08 * max(h, w)) ** 2)))[..., None]
    elif kind == "noise":
        f += rng.normal(0, 6, f.shape)
    else:
        raise ValueError(kind)
    return np.clip(f, 0, 255).astype(np.uint8)


ROBUSTNESS = ("brightness", "dark", "white_balance", "gamma", "shadow", "specular", "noise")


def run_benchmark(approaches: dict[str, Callable[[], Segmenter]], data: Sequence[tuple[str, np.ndarray, np.ndarray]],
                  splits: dict[str, str], cfg: dict, out_dir: str | Path, *, reference_type: str,
                  write_outputs: bool = True, robustness: bool = True) -> dict:
    """data: [(image_id, image, reference_mask)]; splits: image_id -> train/val/test.

    ``reference_type`` must describe the reference honestly: 'adjudicated_expert',
    'single_expert', 'pseudo' or 'synthetic'. Anything but 'adjudicated_expert' is labelled
    provisional in the report.
    """
    out_dir = Path(out_dir)
    ev = cfg["evaluation"]
    train = [(i, im, m) for i, im, m in data if splits[i] == "train"]
    test = [(i, im, m) for i, im, m in data if splits[i] == "test"]
    if not test:
        raise ValueError("no test images")
    # Rare classes fixed from TRAINING frequencies (< 5% of valid pixels), before testing.
    tr_counts = np.bincount(np.concatenate([m[m != UNCERTAIN].ravel() for _, _, m in train]),
                            minlength=NUM_CLASSES) if train else np.ones(NUM_CLASSES)
    frac = tr_counts / max(tr_counts.sum(), 1)
    rare = tuple(int(c) for c in range(1, NUM_CLASSES) if frac[c] < 0.05) or tuple(range(1, NUM_CLASSES))
    report: dict = {"reference_type": reference_type,
                    "provisional": reference_type != "adjudicated_expert",
                    "n_train": len(train), "n_test": len(test), "test_ids": [t[0] for t in test],
                    "rare_classes": [CLASS_NAMES[c] for c in rare], "approaches": {}}
    per_image_primary: dict[str, list[float]] = {}
    rng = np.random.default_rng(cfg.get("seed", 0))
    for name, factory in approaches.items():
        seg = factory()
        t0 = time.perf_counter()
        if seg.requires_training:
            seg.fit([t[1] for t in train], [t[2] for t in train])
        fit_s = time.perf_counter() - t0
        rows, cm_total, times, cov = [], np.zeros((NUM_CLASSES, NUM_CLASSES + 1), np.int64), [], []
        tracemalloc.start()
        for iid, im, ref in test:
            t1 = time.perf_counter()
            mask, unc, proba = seg.predict_mask(im)
            times.append(time.perf_counter() - t1)
            met = evaluate_segmentation(ref, mask, proba if seg.probabilistic else None,
                                        ev["boundary_tolerance_px"], ev["small_region_max_px"], rare)
            met["review_fraction"] = float((mask == UNCERTAIN).mean())
            cm_total += confusion(ref, mask)
            cov.append(coverage_risk(ref, proba))
            rows.append({"image_id": iid, **{k: met.get(k) for k in SUMMARY_KEYS},
                         "per_class_iou": met["per_class_iou"], "review_fraction": met["review_fraction"]})
            if write_outputs:
                save_annotation_outputs(out_dir / name / iid, im, mask, unc, proba, image_id=iid,
                                        approach=seg.describe(), cfg=cfg, review_status="proposed",
                                        evaluation={"reference_type": reference_type,
                                                    **{k: met.get(k) for k in SUMMARY_KEYS}})
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rob = {}
        if robustness:
            for kind in ROBUSTNESS:
                cmr = sum(confusion(ref, seg.predict_mask(perturb(im, kind, rng))[0]) for _, im, ref in test)
                rob[kind] = scores_from_confusion(cmr)["macro_miou"]
        summ = {}
        for k in SUMMARY_KEYS + ("review_fraction",):
            vals = [r[k] for r in rows if r.get(k) is not None]
            m, lo, hi = mean_ci(vals, ev["bootstrap_iterations"], ev["alpha"], cfg.get("seed", 0))
            summ[k] = m
            summ[k + "_ci"] = [lo, hi]
        pooled = scores_from_confusion(cm_total)
        summ.update(seconds_per_image=float(np.mean(times)), fit_seconds=fit_s, peak_python_mem_mb=peak / 2 ** 20,
                    robustness_miou=float(np.nanmean(list(rob.values()))) if rob else float("nan"))
        report["approaches"][name] = {
            "describe": seg.describe(), "summary": summ, "pooled": pooled,
            "pooled_confusion": cm_total.tolist(), "robustness_miou": rob, "per_image": rows,
            "coverage_risk_mean": np.mean(np.asarray(cov), 0).tolist(),
        }
        per_image_primary[name] = [r[ev["primary_metric"]] for r in rows]
    report["statistics"] = compare_approaches({k: np.asarray(v) for k, v in per_image_primary.items()},
                                              ev["bootstrap_iterations"], ev["alpha"], cfg.get("seed", 0))
    report["ranking"] = select_best({k: v["summary"] for k, v in report["approaches"].items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=1, default=float))
    (out_dir / "benchmark.md").write_text(markdown_report(report))
    return report


def _f(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def markdown_report(rep: dict) -> str:
    L = ["# Benchmark report", ""]
    if rep["provisional"]:
        L += [f"> **PROVISIONAL**: reference masks are `{rep['reference_type']}`, not adjudicated expert "
              "annotations. These numbers do not measure real corrosion-segmentation accuracy.", ""]
    L += [f"Train images: {rep['n_train']} | Test images: {rep['n_test']} | Rare classes: "
          f"{', '.join(rep['rare_classes'])}", "",
          "| Approach | mIoU [95% CI] | Corrosion mIoU | Healthy IoU | Binary Dice | Boundary F1 | Rare IoU | "
          "ECE | Abstain | Robust mIoU | s/img |", "|" + "---|" * 11]
    for n, a in rep["approaches"].items():
        s = a["summary"]
        L.append(f"| {n} | {_f(s['macro_miou'])} [{_f(s['macro_miou_ci'][0])}, {_f(s['macro_miou_ci'][1])}] | "
                 f"{_f(s['corrosion_miou'])} | {_f(s['healthy_iou'])} | {_f(s['binary_dice'])} | "
                 f"{_f(s['boundary_f1'])} | {_f(s['rare_class_iou'])} | {_f(s['ece'])} | "
                 f"{_f(s['abstention_rate'])} | {_f(s['robustness_miou'])} | {_f(s['seconds_per_image'], 3)} |")
    L += ["", "## Per-class IoU (pooled over test pixels)", "",
          "| Approach | " + " | ".join(CLASS_NAMES) + " |", "|" + "---|" * (NUM_CLASSES + 1)]
    for n, a in rep["approaches"].items():
        L.append(f"| {n} | " + " | ".join(_f(v) for v in a["pooled"]["per_class_iou"]) + " |")
    L += ["", "## Paired comparisons (primary metric, image-level bootstrap, Holm-corrected)", "",
          "| Pair | Mean diff [95% CI] | p (Holm) | Wilcoxon p | d_z | n |", "|---|---|---|---|---|---|"]
    for k, r in rep["statistics"]["pairwise"].items():
        L.append(f"| {k} | {_f(r['mean_diff'])} [{_f(r['ci'][0])}, {_f(r['ci'][1])}] | {_f(r['p_holm'])} | "
                 f"{_f(r['p_wilcoxon'])} | {_f(r['effect_size_dz'], 2)} | {r['n_images']} |")
    if rep["statistics"]["power_note"]:
        L += ["", f"> {rep['statistics']['power_note']}"]
    L += ["", "## Predeclared weighted-rank selection", ""]
    L += [f"{i + 1}. {n} (score {s:.3f})" for i, (n, s) in enumerate(rep["ranking"])]
    L += ["", "Ranking weights are defined in `cucorr.evaluation.statistics.select_best`; a top rank under "
          "these weights is not a universal winner - inspect per-constraint columns."]
    return "\n".join(L) + "\n"
