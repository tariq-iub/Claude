"""Per-image annotation artefacts: masks, overlays, uncertainty, stats.json, metadata."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .. import __version__
from ..config import config_hash
from ..evaluation.metrics import to_binary
from ..schema import CLASS_NAMES, CORROSION_IDS, NUM_CLASSES, PALETTE, UNCERTAIN
from ..utils.io import array_sha256, write_png
from ..visualization.render import boundary_overlay, color_mask, overlay, uncertainty_map

MASK_CONVENTION = {
    "multiclass_mask.png": f"uint8 class id 0..{NUM_CLASSES - 1}; {UNCERTAIN} = Unknown/Uncertain (ignored)",
    "binary_mask.png": f"uint8 0 = Healthy, 1 = any corrosion class, {UNCERTAIN} = Unknown/Uncertain (ignored)",
    "uncertainty_map.png": "uint8 normalised predictive entropy x 255 (0 certain, 255 max)",
    "probabilities.npy": "float16 (H, W, 6) class probabilities/scores, channel = class id",
}


def image_stats(mask: np.ndarray, uncertainty: np.ndarray | None, proba: np.ndarray | None) -> dict:
    total = int(mask.size)
    valid = mask != UNCERTAIN
    nvalid = int(valid.sum())
    per_class = {}
    for c, n in enumerate(CLASS_NAMES):
        cnt = int((mask == c).sum())
        per_class[n] = {"pixels": cnt, "fraction_of_total": cnt / total,
                        "fraction_of_valid": cnt / nvalid if nvalid else None}
        if proba is not None and cnt:
            per_class[n]["mean_confidence"] = float(proba[..., c][mask == c].mean())
    corr = np.isin(mask, CORROSION_IDS)
    lab, ncomp = ndi.label(corr)
    sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([])
    comps = {}
    for c in CORROSION_IDS:
        _, n = ndi.label(mask == c)
        comps[CLASS_NAMES[c]] = int(n)
    out = {
        "height": int(mask.shape[0]), "width": int(mask.shape[1]),
        "total_pixels": total, "valid_pixels": nvalid,
        "classes": per_class,
        "corroded_pixels": int(corr.sum()),
        "corrosion_fraction_of_total": float(corr.sum() / total),
        "corrosion_fraction_of_valid": float(corr.sum() / nvalid) if nvalid else None,
        "uncertain_pixels": int((~valid).sum()), "uncertain_fraction": float((~valid).sum() / total),
        "corrosion_components": int(ncomp),
        "component_size_px": ({"min": int(sizes.min()), "median": float(np.median(sizes)),
                               "max": int(sizes.max()), "mean": float(sizes.mean())} if ncomp else None),
        "components_per_class": comps,
        "area_units": "pixels (no physical calibration supplied; do not interpret as mm^2)",
    }
    if uncertainty is not None:
        out["uncertainty"] = {"mean": float(np.nanmean(uncertainty)),
                              "p95": float(np.nanpercentile(uncertainty, 95))}
    if proba is not None:
        out["confidence"] = {"mean_max_prob": float(proba.max(-1).mean())}
    return out


def save_annotation_outputs(out_dir: str | Path, image: np.ndarray, mask: np.ndarray,
                            uncertainty: np.ndarray | None, proba: np.ndarray | None, *,
                            image_id: str, approach: dict, cfg: dict, input_hash: str | None = None,
                            review_status: str = "proposed", evaluation: dict | None = None,
                            alpha: float | None = None, save_proba: bool | None = None) -> Path:
    """Write every required artefact for one image. Returns the output directory."""
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match image {image.shape[:2]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = cfg.get("outputs", {})
    alpha = o.get("overlay_alpha", 0.45) if alpha is None else alpha
    write_png(out / "multiclass_mask.png", mask.astype(np.uint8))
    write_png(out / "binary_mask.png", to_binary(mask))
    write_png(out / "color_mask.png", color_mask(mask))
    write_png(out / "overlay.png", overlay(image, mask, alpha))
    if o.get("boundary_overlay", True):
        write_png(out / "boundary_overlay.png", boundary_overlay(image, mask))
    if uncertainty is not None:
        write_png(out / "uncertainty_map.png", uncertainty_map(uncertainty))
    if proba is not None and (o.get("save_probabilities", True) if save_proba is None else save_proba):
        np.save(out / "probabilities.npy", proba.astype(np.float16))
    stats = {"image_id": image_id, **image_stats(mask, uncertainty, proba),
             "approach": approach.get("name"), "model_version": approach.get("version"),
             "config_hash": config_hash(cfg), "review_status": review_status}
    if evaluation is not None:
        stats["evaluation_vs_reference"] = evaluation
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
    meta = {
        "image_id": image_id, "input_sha256": input_hash or array_sha256(image),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "framework_version": __version__, "approach": approach, "config_hash": config_hash(cfg),
        "config": cfg, "classes": {i: n for i, n in enumerate(CLASS_NAMES)},
        "uncertain_value": UNCERTAIN, "palette": {str(k): v for k, v in PALETTE.items()},
        "mask_conventions": MASK_CONVENTION, "review_status": review_status,
        "provenance": "automatic prediction; not human-approved" if review_status == "proposed"
        else f"review status: {review_status}",
    }
    (out / "annotation_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return out
