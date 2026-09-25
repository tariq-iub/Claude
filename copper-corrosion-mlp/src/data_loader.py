"""Load labels.xlsx (expert ground truth) and settings.xlsx (class registry
/ visualization legend) and reconcile them into a single canonical dataset.

Semantic relationship between the two files (established by inspection,
see README.md "Dataset Analysis"):

- settings.xlsx is a *class registry*: one row per class id/title plus a
  `rep_color` and three `refN` RGB swatches. These swatches were found to
  closely track the segmentation-legend colors used elsewhere in the
  project rather than physically observed corrosion appearance (e.g. the
  CuCl `ref2` swatch RGB(209,225,255) is nearly identical to the
  visualization legend for that class), and their Lab-converted values are
  far from the expert-labeled Lab centroids for the same class (see
  README.md). settings.xlsx is therefore used ONLY for the canonical class
  id/name/order and for building the deployment segmentation legend -
  never as a feature-space color prototype.
- labels.xlsx is the authoritative, expert-labeled ground truth: one row
  per observation with CIELAB coordinates and an `annotation_title` label.
  This is the only source used to fit the model and to derive any
  data-driven class prototypes (e.g. for Delta-E00-to-prototype features),
  and prototypes are always computed from the training partition only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import openpyxl

CANONICAL_CLASSES = ["Healthy", "Cu2O", "CuO", "CuCl", "CuCl2"]
IGNORE_CLASS = "Ignore"


@dataclass
class ClassRegistry:
    """Parsed settings.xlsx: class ids/names/order + legend colors only."""

    ids: dict
    names: list
    legend_rgb: dict  # name -> (r,g,b) rep_color, for segmentation overlays only
    reference_swatches: dict  # name -> list[(r,g,b)] refN swatches (legend-only)


@dataclass
class LabeledDataset:
    lab: np.ndarray          # (N,3) float64 CIELAB
    labels: np.ndarray       # (N,) str, one of CANONICAL_CLASSES
    label_ids: np.ndarray    # (N,) int, index into CANONICAL_CLASSES
    row_index: np.ndarray    # (N,) original row number in labels.xlsx (for traceability)
    source_checksum: str = ""
    n_dropped_invalid: int = 0
    quality_report: dict = field(default_factory=dict)


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def load_settings(path: str) -> ClassRegistry:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip().lower() for h in rows[0]]
    idx = {name: header.index(name) for name in ["id", "title", "rep_color", "ref1", "ref2", "ref3"]}

    ids, names, legend_rgb, refs = {}, [], {}, {}
    for r in rows[1:]:
        if r is None or r[idx["id"]] is None:
            continue
        cls_id = int(r[idx["id"]])
        title = str(r[idx["title"]]).strip()
        ids[title] = cls_id
        names.append(title)
        rep = _parse_rgb(r[idx["rep_color"]])
        legend_rgb[title] = rep
        swatches = []
        for key in ("ref1", "ref2", "ref3"):
            v = _parse_rgb(r[idx[key]])
            if v is not None:
                swatches.append(v)
        refs[title] = swatches
    return ClassRegistry(ids=ids, names=names, legend_rgb=legend_rgb, reference_swatches=refs)


def _parse_rgb(cell) -> tuple | None:
    if cell is None:
        return None
    s = str(cell).strip()
    if s.lower() == "nan" or s == "":
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(float(p)) for p in parts)
    except ValueError:
        return None


def load_labels(path: str, invalid_row_policy: str = "remove") -> LabeledDataset:
    """Load labels.xlsx. Validates ranges, drops/errors on invalid rows,
    detects exact and near-duplicates, and reports data-quality diagnostics.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip().lower() for h in rows[0]]
    col = {name: header.index(name) for name in ["lab_l", "lab_a", "lab_b", "annotation_title"]}

    lab_list, label_list, row_idx_list = [], [], []
    n_invalid = 0
    quality = {"missing_value_rows": 0, "unknown_label_rows": 0, "out_of_range_rows": 0}

    for i, r in enumerate(rows[1:], start=2):
        if r is None:
            continue
        L, a, b = r[col["lab_l"]], r[col["lab_a"]], r[col["lab_b"]]
        label = r[col["annotation_title"]]
        if L is None or a is None or b is None or label is None:
            quality["missing_value_rows"] += 1
            n_invalid += 1
            if invalid_row_policy == "error":
                raise ValueError(f"row {i}: missing feature/label value")
            continue
        label = str(label).strip()
        if label == IGNORE_CLASS:
            # Ignore/uncertain rows are excluded from supervised training by
            # design (per settings.xlsx class registry) but are not an error.
            continue
        if label not in CANONICAL_CLASSES:
            quality["unknown_label_rows"] += 1
            n_invalid += 1
            if invalid_row_policy == "error":
                raise ValueError(f"row {i}: unknown label '{label}'")
            continue
        L, a, b = float(L), float(a), float(b)
        if not (0.0 <= L <= 100.0) or not (-128.0 <= a <= 127.0) or not (-128.0 <= b <= 127.0):
            quality["out_of_range_rows"] += 1
            n_invalid += 1
            if invalid_row_policy == "error":
                raise ValueError(f"row {i}: L*a*b* out of physical range")
            continue
        lab_list.append((L, a, b))
        label_list.append(label)
        row_idx_list.append(i)

    lab = np.array(lab_list, dtype=np.float64)
    labels = np.array(label_list, dtype=object)
    label_ids = np.array([CANONICAL_CLASSES.index(l) for l in labels], dtype=np.int64)
    row_index = np.array(row_idx_list, dtype=np.int64)

    # duplicate diagnostics (exact Lab match, rounded to 4 dp)
    keys = [tuple(np.round(x, 4)) for x in lab]
    from collections import defaultdict
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    exact_dupe_groups = {k: v for k, v in groups.items() if len(v) > 1}
    conflicting = 0
    for k, idxs in exact_dupe_groups.items():
        if len(set(labels[idxs])) > 1:
            conflicting += 1
    quality["exact_duplicate_groups"] = len(exact_dupe_groups)
    quality["exact_duplicate_rows"] = sum(len(v) for v in exact_dupe_groups.values())
    quality["conflicting_label_duplicate_groups"] = conflicting
    quality["n_valid"] = len(lab)
    quality["n_invalid_total"] = n_invalid
    quality["pct_invalid"] = 100.0 * n_invalid / max(1, n_invalid + len(lab))

    checksum = _sha256_of_file(path)

    return LabeledDataset(
        lab=lab, labels=labels, label_ids=label_ids, row_index=row_index,
        source_checksum=checksum, n_dropped_invalid=n_invalid, quality_report=quality,
    )
