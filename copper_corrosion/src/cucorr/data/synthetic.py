"""Synthetic copper-like fixtures for tests and demos. NOT real data; never report
results on these as corrosion-segmentation performance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..schema import NUM_CLASSES

# Representative colours taken from settings.xlsx rep_color (class order 0..5).
SYN_COLORS = np.array([(148, 106, 53), (226, 193, 192), (108, 69, 61), (253, 253, 239),
                       (143, 235, 235), (75, 137, 122)], float)


def synthetic_image(h: int = 96, w: int = 128, seed: int = 0, n_blobs: int = 6,
                    noise: float = 8.0, illum: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Healthy background with smooth random corrosion blobs, texture noise and an
    illumination gradient. Returns (rgb uint8, mask uint8)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.uint8)
    for _ in range(n_blobs):
        field = ndi.gaussian_filter(rng.standard_normal((h, w)), rng.uniform(4, 10))
        blob = field > np.quantile(field, rng.uniform(0.88, 0.97))
        mask[blob] = rng.integers(1, NUM_CLASSES)
    img = SYN_COLORS[mask] + rng.normal(0, noise, (h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1 + illum * ((xx / w) - 0.5) * 2
    img = img * grad[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), mask


def synthetic_dataset(n: int = 12, seed: int = 0, **kw) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """List of (image_id, image, mask, group_id). Groups simulate specimens (2 images each)."""
    return [(f"syn_{i:03d}", *synthetic_image(seed=seed + i, **kw), f"specimen_{i // 2:02d}") for i in range(n)]


def write_dataset(root: str | Path, n: int = 12, seed: int = 0) -> Path:
    from ..utils.io import write_png
    root = Path(root)
    rows = ["image_id,group_id,split"]
    for i, (iid, img, m, g) in enumerate(synthetic_dataset(n, seed)):
        write_png(root / "images" / f"{iid}.png", img)
        write_png(root / "masks" / f"{iid}.png", m)
        rows.append(f"{iid},{g},")
    (root / "groups.csv").write_text("\n".join(rows) + "\n")
    return root
