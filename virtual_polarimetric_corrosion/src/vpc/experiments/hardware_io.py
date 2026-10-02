"""Measured-polarimetry pipeline for the hardware-teacher dataset (see docs/HARDWARE_ACQUISITION_PROTOCOL.md).

Input: a folder with one linear (RAW-derived or 16-bit linear TIFF/PNG) image per analyzer angle, plus optional dark and
flat-field frames. Output: calibrated analyzer images, measured Stokes (S0,S1,S2), DoLP, AoLP in a single .npz.
Assumptions that must be satisfied physically: fixed camera/object between angles (sub-pixel registration), no auto-exposure /
auto-white-balance, linear sensor response, polarizer rotation referenced to the image x-axis with a known zero offset.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

import numpy as np


def _load(path: str) -> np.ndarray:
    from PIL import Image
    a = np.asarray(Image.open(path)).astype(np.float64)
    return a / (65535.0 if a.max() > 255 else 255.0)


def load_polarization_stack(folder: str, angles_deg: Sequence[float], pattern: str = "I{angle}.png",
                            dark: Optional[str] = None, flat: Optional[str] = None, zero_offset_deg: float = 0.0) -> Dict[str, np.ndarray]:
    stack = []
    d = _load(dark) if dark else 0.0
    f = _load(flat) if flat else None
    for a in angles_deg:
        name = pattern.format(angle=("%g" % a).replace(".", "_"))
        im = _load(os.path.join(folder, name)) - d
        if f is not None:
            fl = f - d
            im = im / np.clip(fl / fl.mean(), 0.2, None)
        stack.append(np.clip(im, 0, None))
    return {"stack": np.stack(stack), "angles_deg": np.asarray(angles_deg, float) + zero_offset_deg}


def measured_stokes_np(stack: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Least-squares linear Stokes. stack (A,H,W[,C]) -> S (3,H,W[,C])."""
    th = np.radians(angles_deg)
    B = 0.5 * np.stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th)], 1)   # (A,3)
    flat = stack.reshape(len(th), -1)
    S = np.linalg.lstsq(B, flat, rcond=None)[0]
    return S.reshape((3,) + stack.shape[1:])


def process_folder(folder: str, angles_deg: Sequence[float], out_npz: str, **kw) -> Dict[str, np.ndarray]:
    d = load_polarization_stack(folder, angles_deg, **kw)
    S = measured_stokes_np(d["stack"], d["angles_deg"])
    dolp = np.clip(np.hypot(S[1], S[2]) / np.maximum(S[0], 1e-6), 0, 1)
    aolp = 0.5 * np.arctan2(S[2], S[1])
    resid = d["stack"] - 0.5 * (S[0][None] + S[1][None] * np.cos(2 * np.radians(d["angles_deg"])).reshape(-1, *([1] * S[0].ndim))
                                + S[2][None] * np.sin(2 * np.radians(d["angles_deg"])).reshape(-1, *([1] * S[0].ndim)))
    out = {**d, "S": S, "dolp": dolp, "aolp": aolp, "fit_rmse": np.sqrt((resid ** 2).mean(0)) if len(angles_deg) > 3 else np.zeros_like(dolp)}
    np.savez_compressed(out_npz, **out)
    return out
