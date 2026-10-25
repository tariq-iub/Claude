"""Common contract for every segmentation approach.

All approaches consume uint8 RGB images (H, W, 3) and return per-class scores
(H, W, 6) summing to 1. ``predict_mask`` applies the shared abstention rule and
returns (mask uint8 with UNCERTAIN=255, uncertainty float32 in [0, 1]).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..color_science.color import normalize
from ..schema import NUM_CLASSES, UNCERTAIN


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return (-(p * np.log(p)).sum(-1) / np.log(proba.shape[-1])).astype(np.float32)


class Segmenter(ABC):
    name: str = "base"
    version: str = "0.1.0"
    requires_training: bool = True
    #: True when ``predict_proba`` returns calibrated-enough probabilities for ECE/Brier.
    probabilistic: bool = True

    def __init__(self, cfg: dict, abstain_confidence: float = 0.0):
        self.cfg = cfg
        self.abstain_confidence = abstain_confidence

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB (H,W,3), got {image.dtype} {image.shape}")
        return normalize(image, self.cfg.get("color", {}))

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> "Segmenter":
        """Train on images and class-index masks (UNCERTAIN pixels are ignored)."""
        return self

    @abstractmethod
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Return (H, W, NUM_CLASSES) float32 probabilities."""

    def abstain(self, proba: np.ndarray) -> np.ndarray:
        """Boolean (H, W) mask of pixels on which to abstain."""
        return proba.max(-1) < self.abstain_confidence

    def predict_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask, uncertainty, proba)."""
        proba = self.predict_proba(image)
        assert proba.shape == (*image.shape[:2], NUM_CLASSES), proba.shape
        mask = proba.argmax(-1).astype(np.uint8)
        mask[self.abstain(proba)] = UNCERTAIN
        return mask, normalized_entropy(proba), proba

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "class": type(self).__name__}


def valid_pixels(mask: np.ndarray) -> np.ndarray:
    return mask != UNCERTAIN
