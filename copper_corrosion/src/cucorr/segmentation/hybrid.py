"""Approach F: hybrid ensemble used inside the active-learning loop.

Components (configurable) are combined by averaging probabilities. Disagreement between
components (mean pairwise total-variation distance) is exposed for review selection.
Approach A provides initial pseudo-labels when no reviewed masks exist yet.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schema import UNCERTAIN
from .base import Segmenter


class HybridSegmenter(Segmenter):
    name = "approach_f_hybrid"

    def __init__(self, cfg: dict, components: Sequence[Segmenter], abstain_confidence: float = 0.4):
        super().__init__(cfg, abstain_confidence)
        self.components = list(components)

    def fit(self, images, masks) -> "HybridSegmenter":
        for c in self.components:
            if c.requires_training:
                c.fit(images, masks)
        return self

    def component_probas(self, image: np.ndarray) -> list[np.ndarray]:
        return [c.predict_proba(image) for c in self.components]

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        ps = self.component_probas(image)
        self.last_disagreement = disagreement(ps)
        return np.mean(ps, 0).astype(np.float32)

    def describe(self) -> dict:
        d = super().describe()
        d["components"] = [c.describe() for c in self.components]
        return d


def disagreement(probas: Sequence[np.ndarray]) -> np.ndarray:
    """Mean pairwise total-variation distance between component predictions, in [0, 1]."""
    if len(probas) < 2:
        return np.zeros(probas[0].shape[:2], np.float32)
    acc, n = 0.0, 0
    for i in range(len(probas)):
        for j in range(i + 1, len(probas)):
            acc = acc + 0.5 * np.abs(probas[i] - probas[j]).sum(-1)
            n += 1
    return (acc / n).astype(np.float32)


def pseudo_labels(seg: Segmenter, image: np.ndarray) -> np.ndarray:
    """Abstaining predictions usable as weak training targets (UNCERTAIN ignored)."""
    m, _, _ = seg.predict_mask(image)
    return m
