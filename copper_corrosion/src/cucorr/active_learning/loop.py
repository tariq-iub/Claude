"""Active learning: choose images for expert review and retrain iteratively.

The *oracle* supplies reviewed masks. In simulations the oracle returns stored reference
masks; in production it is the annotation app (reviewed masks from AnnotationStore).
Strategies are compared at equal annotation budgets (images reviewed per round).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..evaluation.metrics import confusion, scores_from_confusion
from ..segmentation.base import Segmenter, normalized_entropy
from ..segmentation.hybrid import disagreement


def score_images(model: Segmenter, images: Sequence[np.ndarray], strategy: str,
                 committee: Sequence[Segmenter] | None = None, rng=None) -> np.ndarray:
    if strategy == "random":
        return (rng or np.random.default_rng(0)).random(len(images))
    out = []
    for im in images:
        if strategy == "uncertainty":
            out.append(float(normalized_entropy(model.predict_proba(im)).mean()))
        elif strategy == "disagreement":
            if not committee or len(committee) < 2:
                raise ValueError("disagreement strategy needs >= 2 committee models")
            out.append(float(disagreement([c.predict_proba(im) for c in committee]).mean()))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    return np.asarray(out)


def evaluate(model: Segmenter, images, masks) -> float:
    cm = sum(confusion(m, model.predict_mask(i)[0]) for i, m in zip(images, masks))
    return scores_from_confusion(cm)["macro_miou"]


def run_active_learning(make_model: Callable[[], Segmenter], pool_images: Sequence[np.ndarray],
                        oracle: Callable[[int], np.ndarray], val_images, val_masks, *,
                        strategy: str, budget_per_round: int, max_rounds: int, init_idx: Sequence[int],
                        min_improvement: float = 0.005, patience: int = 2, seed: int = 0,
                        make_committee: Callable[[], list[Segmenter]] | None = None) -> dict:
    """Return the learning curve [(n_reviewed, val_mIoU), ...] and selected indices."""
    rng = np.random.default_rng(seed)
    labelled = list(init_idx)
    curve, stall = [], 0
    for rnd in range(max_rounds + 1):
        model = make_model().fit([pool_images[i] for i in labelled], [oracle(i) for i in labelled])
        score = evaluate(model, val_images, val_masks)
        curve.append((len(labelled), score))
        if rnd > 0 and score - curve[-2][1] < min_improvement:
            stall += 1
        else:
            stall = 0
        unl = [i for i in range(len(pool_images)) if i not in labelled]
        if rnd == max_rounds or not unl or stall >= patience:
            break
        committee = None
        if strategy == "disagreement" and make_committee:
            committee = [c.fit([pool_images[i] for i in labelled], [oracle(i) for i in labelled])
                         if c.requires_training else c for c in make_committee()]
        s = score_images(model, [pool_images[i] for i in unl], strategy, committee, rng)
        labelled += [unl[j] for j in np.argsort(-s)[:budget_per_round]]
    return {"strategy": strategy, "curve": curve, "labelled": labelled,
            "stopped_reason": "max_rounds" if len(curve) == max_rounds + 1 else
            ("diminishing_improvement" if stall >= patience else "pool_exhausted")}
