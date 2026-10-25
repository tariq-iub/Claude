"""Approach E: promptable foundation-model segmentation (SAM 2) + class assignment.

SAM 2 proposes class-agnostic regions; each region receives the majority (probability-
averaged) class of a colour/pixel classifier (``label_assigner``). Boundary quality and
label quality are therefore evaluated separately (see evaluation.metrics.boundary_f1 on
the binary mask vs. per-class IoU).

Prerequisites (not bundled): ``pip install 'cucorr[sam]'`` and a checkpoint from
https://github.com/facebookresearch/sam2 (Apache-2.0 code/weights at time of writing -
re-verify the licence of the exact checkpoint before redistribution). A GPU is strongly
recommended; CPU inference works but is slow.

When SAM 2 is not installed, ``region_proposer`` may be set to ``"felzenszwalb"`` to run
the identical pipeline with a classical graph-based proposer - results from that mode
must NOT be reported as foundation-model results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.segmentation import felzenszwalb

from ..schema import NUM_CLASSES
from .base import Segmenter


class FoundationSegmenter(Segmenter):
    name = "approach_e_foundation"

    def __init__(self, cfg: dict, label_assigner: Segmenter, region_proposer: str = "sam2"):
        super().__init__(cfg)
        self.p = cfg["approach_e"]
        self.assigner = label_assigner
        self.proposer = region_proposer
        self.probabilistic = label_assigner.probabilistic
        self._gen = None
        if region_proposer == "sam2":
            self._load_sam2()

    def _load_sam2(self):
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise ImportError("SAM 2 not installed: pip install 'cucorr[sam]' (see module docstring)") from e
        ckpt = Path(self.p["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM 2 checkpoint not found at {ckpt}; download it from the official repo")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(self.p["model_cfg"], str(ckpt), device=dev)
        self._gen = SAM2AutomaticMaskGenerator(model, points_per_side=self.p["points_per_side"])

    def propose_regions(self, image: np.ndarray) -> np.ndarray:
        """Return an int label image; 0..K-1 are regions (unassigned pixels get their own label)."""
        if self.proposer == "felzenszwalb":
            return felzenszwalb(image, scale=200, sigma=0.8, min_size=50)
        masks = sorted(self._gen.generate(image), key=lambda m: -m["area"])
        lab = np.full(image.shape[:2], -1, int)
        for i, m in enumerate(masks):  # small masks drawn last win overlaps
            lab[m["segmentation"]] = i
        lab[lab < 0] = len(masks)
        return lab

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        regions = self.propose_regions(image)
        pix = self.assigner.predict_proba(image)
        n = regions.max() + 1
        flat = regions.ravel()
        cnt = np.maximum(np.bincount(flat, minlength=n), 1)
        avg = np.stack([np.bincount(flat, pix[..., c].ravel(), n) / cnt for c in range(NUM_CLASSES)], 1)
        self.last_regions = regions
        return avg[regions].astype(np.float32)

    def describe(self) -> dict:
        d = super().describe()
        d.update(region_proposer=self.proposer, label_assigner=self.assigner.name,
                 checkpoint=self.p["checkpoint"] if self.proposer == "sam2" else None,
                 is_foundation_model=self.proposer == "sam2")
        return d
