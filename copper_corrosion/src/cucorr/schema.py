"""Stable class schema shared by every module.

Class IDs are fixed. ``UNCERTAIN`` is a workflow state, never a corrosion class.
Visualization colours are deliberately distinct from the expert reference colours
in ``settings.xlsx`` (those are metadata, see :mod:`cucorr.data.workbooks`).
"""
from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("Healthy", "Cu2O", "CuO", "CuCl", "CuCl2", "Patina")
NUM_CLASSES = len(CLASS_NAMES)
HEALTHY = 0
CORROSION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
UNCERTAIN = 255          # multiclass mask value for Unknown/Uncertain / ignored pixels
BINARY_UNCERTAIN = 255   # binary mask value for Unknown/Uncertain

# High-contrast visualization palette (Okabe-Ito derived, colour-blind friendly).
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),   # Healthy - light grey
    1: (230, 159, 0),     # Cu2O - orange
    2: (0, 0, 0),         # CuO - black
    3: (86, 180, 233),    # CuCl - sky blue
    4: (0, 114, 178),     # CuCl2 - blue
    5: (0, 158, 115),     # Patina - green
    UNCERTAIN: (204, 121, 167),  # Uncertain - magenta
}


def class_id(name: str) -> int:
    """Map an exact class name to its ID (raises ``KeyError`` on unknown names)."""
    norm = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    key = name.strip().lower()
    if key not in norm:
        raise KeyError(f"Unknown class name {name!r}; expected one of {CLASS_NAMES}")
    return norm[key]
