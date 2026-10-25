"""Image I/O helpers (lossless only for masks)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB (H, W, 3). EXIF orientation is NOT applied, to keep
    pixel alignment with the stored file."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"{path}: class-index masks must be single-channel, got {arr.shape}")
    return arr.astype(np.uint8)


def write_png(path: str | Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG", optimize=False)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def list_images(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
