"""Factory that builds any approach from configuration."""
from __future__ import annotations

from ..config import resolve
from ..data.workbooks import load_manifest, load_settings, prototypes
from .base import Segmenter
from .color_match import ColorMatchSegmenter
from .pixel_ml import PixelClassifierSegmenter
from .superpixel import SuperpixelSegmenter


def build(name: str, cfg: dict) -> Segmenter:
    if name == "approach_a":
        s = load_settings(resolve(cfg, "settings_xlsx"), cfg["classes"]["settings_id_offset"])
        m = load_manifest(resolve(cfg, "labels_xlsx"))
        lab, rgb, ids = prototypes(s, m, cfg["approach_a"]["use_manifest_prototypes"])
        return ColorMatchSegmenter(cfg, lab, rgb, ids)
    if name == "approach_b":
        return PixelClassifierSegmenter(cfg)
    if name == "approach_c":
        return SuperpixelSegmenter(cfg)
    if name == "approach_d":
        from .deep import DeepSegmenter
        return DeepSegmenter(cfg)
    if name == "approach_e":
        from .foundation import FoundationSegmenter
        return FoundationSegmenter(cfg, build(cfg["approach_e"]["label_assigner"], cfg),
                                   cfg["approach_e"].get("region_proposer", "sam2"))
    if name == "approach_f":
        from .hybrid import HybridSegmenter
        return HybridSegmenter(cfg, [build(c, cfg) for c in cfg["approach_f"]["components"]])
    raise KeyError(f"unknown approach {name!r}")
