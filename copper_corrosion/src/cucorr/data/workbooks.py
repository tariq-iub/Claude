"""Parsing and auditing of ``settings.xlsx`` and ``labels.xlsx``.

Interpretation (established by audit, see docs/data_dictionary.md):

* ``settings.xlsx`` / sheet ``settings``: one row per class. ``rep_color`` is the
  representative sRGB colour; ``ref1..ref3`` are *additional* example colours of the
  same class (multiple prototypes), not alternatives to be chosen between.
  Values are ``"R,G,B"`` strings, sometimes with a trailing newline.
* ``labels.xlsx`` / sheet ``manifest``: one row per *colour cluster* (``cluster_id``)
  with its CIELAB centroid and an expert-assigned class title. It is **not** an image
  index and **not** pixel-level ground truth: it contains no image or pixel reference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..color_science.color import delta_e, lab_to_rgb, rgb_to_lab
from ..schema import CLASS_NAMES, class_id

RGB_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$")
REF_COLS = ("rep_color", "ref1", "ref2", "ref3")


def parse_rgb(value) -> tuple[int, int, int] | None:
    """Parse ``"R,G,B"`` (whitespace/newlines tolerated). Missing -> ``None``.
    Malformed or out-of-range values raise ``ValueError`` rather than being guessed."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    m = RGB_RE.match(s)
    if not m:
        raise ValueError(f"Malformed RGB value {value!r}")
    rgb = tuple(int(x) for x in m.groups())
    if any(c > 255 for c in rgb):
        raise ValueError(f"RGB component out of range in {value!r}")
    return rgb  # type: ignore[return-value]


@dataclass
class ClassReference:
    class_id: int
    name: str
    settings_id: int
    rep_color: tuple[int, int, int] | None
    refs: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def all_colors(self) -> list[tuple[int, int, int]]:
        return ([self.rep_color] if self.rep_color else []) + self.refs


def load_settings(path: str | Path, id_offset: int = 1) -> list[ClassReference]:
    df = pd.read_excel(path, sheet_name="settings")
    out = []
    for _, row in df.iterrows():
        name = str(row["title"]).strip()
        cid = class_id(name)
        if int(row["id"]) - id_offset != cid:
            raise ValueError(f"settings id {row['id']} for {name} does not map to class {cid}")
        refs = [parse_rgb(row[c]) for c in REF_COLS[1:] if c in df.columns]
        out.append(ClassReference(cid, CLASS_NAMES[cid], int(row["id"]), parse_rgb(row["rep_color"]),
                                  [r for r in refs if r is not None]))
    out.sort(key=lambda r: r.class_id)
    if [r.class_id for r in out] != list(range(len(CLASS_NAMES))):
        raise ValueError("settings.xlsx must define each of the six classes exactly once")
    return out


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="manifest")
    required = {"cluster_id", "lab_l", "lab_a", "lab_b", "annotation_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns {missing}")
    df = df.copy()
    df["class_id"] = df["annotation_title"].map(class_id)
    return df


def prototypes(settings: list[ClassReference], manifest: pd.DataFrame | None = None,
               use_manifest: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lab[N,3], rgb[N,3], class_ids[N]) for every reference colour.
    Manifest Lab centroids are converted to RGB only for the RGB baseline."""
    labs, rgbs, ids = [], [], []
    for ref in settings:
        for c in ref.all_colors:
            rgbs.append(c)
            labs.append(rgb_to_lab(np.array(c, np.uint8)))
            ids.append(ref.class_id)
    if use_manifest and manifest is not None:
        lab = manifest[["lab_l", "lab_a", "lab_b"]].to_numpy(float)
        labs.extend(lab)
        rgbs.extend(lab_to_rgb(lab))
        ids.extend(manifest["class_id"].to_numpy())
    return np.asarray(labs, float), np.asarray(rgbs, float), np.asarray(ids, int)


def audit(settings_path: str | Path, labels_path: str | Path) -> dict:
    """Produce a machine-readable audit of both workbooks."""
    report: dict = {"warnings": [], "assumptions": [], "unresolved": []}
    s_book = pd.read_excel(settings_path, sheet_name=None)
    l_book = pd.read_excel(labels_path, sheet_name=None)
    report["sheets"] = {
        "settings.xlsx": {k: {"shape": list(v.shape), "columns": {c: str(t) for c, t in v.dtypes.items()},
                              "missing": v.isna().sum().astype(int).to_dict()} for k, v in s_book.items()},
        "labels.xlsx": {k: {"shape": list(v.shape), "columns": {c: str(t) for c, t in v.dtypes.items()},
                            "missing": v.isna().sum().astype(int).to_dict()} for k, v in l_book.items()},
    }
    raw = s_book["settings"]
    trailing = int(sum(str(v) != str(v).strip() for c in REF_COLS for v in raw[c]))
    if trailing:
        report["warnings"].append(f"{trailing} RGB cells contain trailing whitespace/newlines (stripped on parse).")
    settings = load_settings(settings_path)
    report["classes"] = []
    for ref in settings:
        labs = [rgb_to_lab(np.array(c, np.uint8)).round(2).tolist() for c in ref.all_colors]
        report["classes"].append({"class_id": ref.class_id, "name": ref.name, "settings_id": ref.settings_id,
                                  "rep_color": ref.rep_color, "refs": ref.refs, "lab_D65": labs})
    # Outlier references: far from the class's own rep colour relative to other classes.
    for ref in settings:
        rep = rgb_to_lab(np.array(ref.rep_color, np.uint8))
        for c in ref.refs:
            d = float(delta_e(rgb_to_lab(np.array(c, np.uint8)), rep))
            if d > 40:
                report["warnings"].append(f"{ref.name}: reference {c} is dE2000={d:.1f} from rep_color "
                                          f"{ref.rep_color}; verify it is not a mis-entry.")

    m = load_manifest(labels_path)
    report["manifest"] = {
        "rows": int(len(m)),
        "versions": m["version"].value_counts().to_dict() if "version" in m else None,
        "class_counts": m["annotation_title"].value_counts().to_dict(),
        "cluster_id_range": [int(m.cluster_id.min()), int(m.cluster_id.max())],
        "cluster_id_duplicates": int(m.cluster_id.duplicated().sum()),
        "cluster_id_gaps": int(m.cluster_id.max() - m.cluster_id.min() + 1 - m.cluster_id.nunique()),
        "duplicate_lab_rows": int(m.duplicated(["lab_l", "lab_a", "lab_b"]).sum()),
        "lab_ranges": {c: [float(m[c].min()), float(m[c].max())] for c in ("lab_l", "lab_a", "lab_b")},
        "per_class_lab_mean": m.groupby("annotation_title")[["lab_l", "lab_a", "lab_b"]].mean().round(2)
                               .to_dict(orient="index"),
        "extra_columns": [c for c in m.columns if c not in
                          {"cluster_id", "lab_l", "lab_a", "lab_b", "annotation_title", "class_id"}],
    }
    if report["manifest"]["cluster_id_gaps"]:
        report["warnings"].append(f"{report['manifest']['cluster_id_gaps']} cluster_ids absent from 1..max; "
                                  "clusters may have been deleted or filtered (version 3 only).")
    # Out-of-gamut check: Lab centroid that cannot be represented in sRGB.
    lab = m[["lab_l", "lab_a", "lab_b"]].to_numpy(float)
    rt = rgb_to_lab(lab_to_rgb(lab))
    oog = int((delta_e(lab, rt) > 2.0).sum())
    report["manifest"]["out_of_srgb_gamut_rows"] = oog
    # Cross-file consistency: classify manifest centroids with settings references only.
    plab, _, pid = prototypes(settings, None, use_manifest=False)
    d = delta_e(lab[:, None, :], plab[None, :, :])
    pred = pid[d.argmin(1)]
    agree = float((pred == m["class_id"].to_numpy()).mean())
    conf = pd.crosstab(m["annotation_title"], [CLASS_NAMES[i] for i in pred],
                       rownames=["manifest"], colnames=["nearest_settings_ref"])
    report["consistency"] = {"manifest_vs_settings_nearest_ref_agreement": round(agree, 4),
                             "confusion": conf.to_dict()}
    report["class_names_compatible"] = sorted(set(m["annotation_title"])) == sorted(CLASS_NAMES)
    report["assumptions"] += [
        "A1: Workbook RGB and image pixels are sRGB; CIELAB is D65 / 2-degree (skimage). Manifest Lab is "
        "assumed to use the same convention; not documented in the workbook.",
        "A2: settings.xlsx id k maps to class id k-1; class names match exactly between files.",
        "A3: rep_color and ref1..ref3 are all valid prototypes of the same class (multi-prototype).",
        "A4: manifest rows are expert-labelled colour-cluster centroids (palette-level weak supervision), "
        "not image identifiers and not pixel ground truth.",
    ]
    report["unresolved"] += [
        "Which images/pixels each cluster_id was derived from (no image reference in manifest).",
        "Clustering algorithm, colour space and white point used to compute the manifest Lab values.",
        "Meaning of `version` (all rows = 3) and NAVICAT_ROWID (database export key; ignored).",
        "Camera, lighting, calibration (px/mm) and specimen/session identifiers for the ~500 images.",
        "Existence of any dense expert masks; none were supplied.",
        "Whether class labels are supported by chemical characterisation (XRD/Raman/XPS) for any specimen.",
    ]
    if agree < 0.8:
        report["warnings"].append(f"Only {agree:.1%} of manifest centroids match the class of their nearest "
                                  "settings reference colour; the two sources are not interchangeable and "
                                  "classes overlap in colour space.")
    return report
