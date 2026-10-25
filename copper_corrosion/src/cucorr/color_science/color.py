"""Colour-space conversion, colour difference and illumination normalisation.

Assumption A1 (configurable, documented in docs/data_dictionary.md): image pixels and
workbook RGB values are sRGB-encoded; CIELAB uses the D65 white point and the CIE 1931
2-degree observer, as implemented by ``skimage.color.rgb2lab``.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color as skcolor


def rgb_to_lab(rgb: np.ndarray, illuminant: str = "D65", observer: str = "2") -> np.ndarray:
    """uint8 or float [0,1] RGB (..., 3) -> CIELAB float64 (..., 3)."""
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8 or arr.max() > 1.0:
        arr = arr.astype(np.float64) / 255.0
    shape = arr.shape
    lab = skcolor.rgb2lab(arr.reshape(-1, 1, 3), illuminant=illuminant, observer=observer)
    return lab.reshape(shape)


def lab_to_rgb(lab: np.ndarray, illuminant: str = "D65", observer: str = "2") -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    shape = lab.shape
    rgb = skcolor.lab2rgb(lab.reshape(-1, 1, 3), illuminant=illuminant, observer=observer)
    return np.clip(np.round(rgb.reshape(shape) * 255), 0, 255).astype(np.uint8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb).astype(np.float64) / 255.0
    shape = arr.shape
    return skcolor.rgb2hsv(arr.reshape(-1, 1, 3)).reshape(shape)


def delta_e(lab1: np.ndarray, lab2: np.ndarray, method: str = "ciede2000") -> np.ndarray:
    """Colour difference between broadcastable CIELAB arrays."""
    if method == "cie76" or method == "euclidean":
        return np.linalg.norm(np.asarray(lab1) - np.asarray(lab2), axis=-1)
    if method == "ciede2000":
        a, b = np.broadcast_arrays(np.asarray(lab1, float), np.asarray(lab2, float))
        return skcolor.deltaE_ciede2000(a, b)
    raise ValueError(f"Unknown colour-difference method {method!r}")


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Gray-world white balance."""
    f = rgb.astype(np.float64)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def max_rgb(rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """White-patch (robust max-RGB) white balance."""
    f = rgb.astype(np.float64)
    ref = np.percentile(f.reshape(-1, 3), percentile, axis=0) + 1e-6
    return np.clip(f * (255.0 / ref), 0, 255).astype(np.uint8)


def illumination_correct(rgb: np.ndarray, sigma: float = 50) -> np.ndarray:
    """Flatten slow illumination gradients by dividing L* by its Gaussian-blurred field."""
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    field = gaussian_filter(L, sigma) + 1e-6
    lab[..., 0] = np.clip(L * (L.mean() / field), 0, 100)
    return lab_to_rgb(lab)


def normalize(rgb: np.ndarray, cfg_color: dict) -> np.ndarray:
    out = rgb
    if cfg_color.get("illumination_correction"):
        out = illumination_correct(out, cfg_color.get("illumination_sigma", 50))
    method = cfg_color.get("normalization", "none")
    if method == "gray_world":
        out = gray_world(out)
    elif method == "max_rgb":
        out = max_rgb(out)
    elif method != "none":
        raise ValueError(f"Unknown normalization {method!r}")
    return out
