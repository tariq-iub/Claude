"""Approach A: expert-reference colour matching (nearest prototype with abstention)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..color_science.color import delta_e, rgb_to_lab
from ..schema import NUM_CLASSES
from .base import Segmenter


class ColorMatchSegmenter(Segmenter):
    """Assign each pixel to the class of its nearest reference colour.

    * ``color_space='rgb'`` + ``distance='euclidean'`` is the naive RGB baseline.
    * ``color_space='lab'`` with ``cie76`` or ``ciede2000`` uses perceptual distances.
    * ``aggregation='nearest'``: class distance = min over that class's prototypes;
      ``'mean'``: distance to the class's mean prototype.
    * Abstain if best distance > ``max_distance`` or (second - best) < ``abstain_margin``.

    Thresholds may be tuned with :meth:`fit` on *training/validation* masks only.
    """

    name = "approach_a_color_match"
    requires_training = False
    probabilistic = False  # softmax over distances is a score, not a calibrated probability

    def __init__(self, cfg: dict, proto_lab: np.ndarray, proto_rgb: np.ndarray, proto_ids: np.ndarray):
        a = cfg["approach_a"]
        super().__init__(cfg)
        self._lut = None
        self.p = a
        self.proto_lab, self.proto_rgb, self.proto_ids = proto_lab, proto_rgb, proto_ids
        if a["aggregation"] == "mean":
            ids = np.arange(NUM_CLASSES)
            self.proto_lab = np.stack([proto_lab[proto_ids == i].mean(0) for i in ids])
            self.proto_rgb = np.stack([proto_rgb[proto_ids == i].mean(0) for i in ids])
            self.proto_ids = ids

    def class_distances(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 6) min distance to each class's prototypes.

        Distances are cached in a lazily filled lookup table over RGB quantised to
        ``lut_bits`` bits/channel (default 7: step 2, max error 1 RGB unit), so the cost per
        image is bounded by the number of *new* distinct colours rather than pixels.
        """
        img = self.preprocess(image)
        h, w = img.shape[:2]
        bits = int(self.p.get("lut_bits", 7))
        shift = 8 - bits
        q = (img >> shift).astype(np.int64)
        codes = (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]
        if getattr(self, "_lut", None) is None or self._lut.shape[0] != 1 << (3 * bits):
            self._lut = np.full((1 << (3 * bits), NUM_CLASSES), np.nan, np.float32)
        uniq = np.unique(codes)
        todo = uniq[np.isnan(self._lut[uniq, 0])]
        if len(todo):
            centre = (1 << shift) // 2 if shift else 0
            rgb = np.stack([(todo >> (2 * bits)) & ((1 << bits) - 1), (todo >> bits) & ((1 << bits) - 1),
                            todo & ((1 << bits) - 1)], 1)
            rgb = np.clip((rgb << shift) + centre, 0, 255).astype(np.uint8)
            if self.p["color_space"] == "rgb":
                x, protos, method = rgb.astype(float), self.proto_rgb, "euclidean"
            else:
                x, protos, method = rgb_to_lab(rgb), self.proto_lab, self.p["distance"]
            for s in range(0, len(x), 4096):
                dd = delta_e(x[s:s + 4096, None, :], protos[None, :, :], method)
                block = np.full((len(dd), NUM_CLASSES), np.inf, np.float32)
                for c in range(NUM_CLASSES):
                    sel = self.proto_ids == c
                    if sel.any():
                        block[:, c] = dd[:, sel].min(1)
                self._lut[todo[s:s + 4096]] = block
        return self._lut[codes]

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        d = self.class_distances(image)
        self._last_d = d
        z = -d / self.p["softmax_temperature"]
        z -= z.max(-1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(-1, keepdims=True)).astype(np.float32)

    def abstain(self, proba: np.ndarray) -> np.ndarray:
        d = np.sort(self._last_d, axis=-1)
        return (d[..., 0] > self.p["max_distance"]) | ((d[..., 1] - d[..., 0]) < self.p["abstain_margin"])

    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
            margins=(0.0, 1.0, 2.0, 4.0, 8.0), max_ds=(15.0, 25.0, 40.0, 1e9)) -> "ColorMatchSegmenter":
        """Grid-search abstention thresholds maximising accuracy x coverage on TRAIN masks."""
        if not images:
            return self
        from ..schema import UNCERTAIN
        best, cache = (-1.0, None), [(self.class_distances(im), m) for im, m in zip(images, masks)]
        for mg in margins:
            for md in max_ds:
                cor = tot = cov = 0
                for d, m in cache:
                    v = m != UNCERTAIN
                    s = np.sort(d, -1)
                    keep = v & ~((s[..., 0] > md) | (s[..., 1] - s[..., 0] < mg))
                    cor += (d.argmin(-1)[keep] == m[keep]).sum()
                    cov += keep.sum()
                    tot += v.sum()
                score = (cor / max(cov, 1)) * (cov / max(tot, 1)) ** 0.5
                if score > best[0]:
                    best = (score, (mg, md))
        self.p = {**self.p, "abstain_margin": best[1][0], "max_distance": best[1][1]}
        return self
