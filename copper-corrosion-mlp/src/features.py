"""Candidate feature representations built on top of raw CIELAB.

labels.xlsx only stores Lab, so HSV/chroma/intensity are *derived*, not
independently measured -> they cannot add information that isn't already
a deterministic function of L*a*b*. They are included as *ablation
candidates* (a nonlinear re-parameterization can still help a shallow MLP
separate classes) rather than assumed-beneficial inputs; §6/§20 of the
task brief require this to be verified empirically, not assumed. See
results/feature_ablation.csv.
"""
from __future__ import annotations

import numpy as np

from . import color_metrics as cm

FEATURE_SETS = ["lab", "lab_hsv", "extended", "lab_deltaE", "extended_deltaE"]


def lab_to_approx_rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse CIELAB->sRGB (D65), used only to derive HSV features from Lab
    (labels.xlsx has no native RGB/HSV column)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def finv(t):
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0 / 29.0))

    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = np.stack([finv(fx) * white[0], finv(fy) * white[1], finv(fz) * white[2]], axis=-1)
    m_inv = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]]
    )
    lin = xyz @ m_inv.T
    a_ = 0.055
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, (1 + a_) * np.clip(lin, 0, None) ** (1 / 2.4) - a_)
    srgb = np.clip(srgb, 0.0, 1.0)
    return srgb * 255.0


def circular_hue(h_fraction: np.ndarray) -> np.ndarray:
    """H in [0,1) -> (sin(2*pi*H), cos(2*pi*H)) to avoid the 0/1 wrap discontinuity."""
    ang = 2 * np.pi * h_fraction
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1)


def build_prototypes(lab_train: np.ndarray, labels_train: np.ndarray, class_names: list) -> np.ndarray:
    """Per-class Lab centroid computed from the TRAINING partition only.
    Used for Delta-E00-to-prototype features. Never fit on val/test."""
    protos = []
    for c in class_names:
        m = labels_train == c
        if m.sum() == 0:
            raise ValueError(f"no training samples for class {c}; cannot build prototype")
        protos.append(lab_train[m].mean(axis=0))
    return np.stack(protos, axis=0)


def build_features(lab: np.ndarray, feature_set: str, prototypes: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) for the requested feature_set.

    lab: (N,3) CIELAB. prototypes: (K,3) required for '*_deltaE' sets,
    must be derived from the training partition only (see build_prototypes).
    """
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = cm.lab_chroma(lab)
    parts = [lab]
    names = ["L", "a", "b"]

    if feature_set in ("lab_hsv", "extended", "extended_deltaE"):
        rgb_approx = lab_to_approx_rgb(lab)
        hsv = cm.rgb_to_hsv(rgb_approx)
        hs_hc = circular_hue(hsv[:, 0])
        parts += [hs_hc, hsv[:, 1:2], hsv[:, 2:3]]
        names += ["H_sin", "H_cos", "S", "V"]

    if feature_set in ("extended", "extended_deltaE"):
        intensity = lab[:, 0] / 100.0  # normalized luminance proxy
        parts += [chroma[:, None], intensity[:, None]]
        names += ["C", "I"]

    if feature_set in ("lab_deltaE", "extended_deltaE"):
        if prototypes is None:
            raise ValueError("prototypes required for a Delta-E00 feature set")
        d = cm.ciede2000(lab, prototypes)  # (N, K)
        parts.append(d)
        names += [f"dE00_{i}" for i in range(prototypes.shape[0])]

    X = np.concatenate(parts, axis=1)
    return X.astype(np.float64), names
