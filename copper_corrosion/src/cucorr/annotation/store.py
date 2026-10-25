"""Append-only annotation store: proposals, human revisions, blinded independent labels,
and adjudication. Nothing is ever overwritten; every revision is a new versioned file.

Layout::

    <root>/<image_id>/proposals/<approach>.png
    <root>/<image_id>/revisions/<annotator>/<NNNN>.png   (+ .json sidecar)
    <root>/<image_id>/adjudicated/<NNNN>.png             (+ .json sidecar)
    <root>/audit_log.jsonl
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from ..evaluation.agreement import disagreement_map
from ..schema import UNCERTAIN
from ..utils.io import array_sha256, read_mask, write_png

STATUSES = ("proposed", "accepted", "rejected", "corrected", "independent", "adjudicated")


class AnnotationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _log(self, **event):
        event["utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with open(self.root / "audit_log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    def save_proposal(self, image_id: str, approach: str, mask: np.ndarray) -> Path:
        p = self.root / image_id / "proposals" / f"{approach}.png"
        if p.exists():
            p = p.with_name(f"{approach}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}.png")
        write_png(p, mask)
        self._log(action="proposal", image_id=image_id, approach=approach, path=str(p), sha=array_sha256(mask))
        return p

    def _next(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.png")))
        return d / f"{n:04d}.png"

    def save_revision(self, image_id: str, annotator: str, mask: np.ndarray, status: str,
                      source: str | None = None, note: str = "", seconds_spent: float | None = None,
                      blinded: bool = False) -> Path:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        p = self._next(self.root / image_id / "revisions" / annotator)
        write_png(p, mask)
        meta = {"image_id": image_id, "annotator": annotator, "status": status, "source": source,
                "note": note, "seconds_spent": seconds_spent, "blinded": blinded, "sha": array_sha256(mask)}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self._log(action="revision", path=str(p), **meta)
        return p

    def latest(self, image_id: str, annotator: str) -> np.ndarray | None:
        d = self.root / image_id / "revisions" / annotator
        files = sorted(d.glob("*.png")) if d.exists() else []
        return read_mask(files[-1]) if files else None

    def independent_labels(self, image_id: str) -> dict[str, np.ndarray]:
        """Latest *blinded* label per annotator (used for agreement)."""
        out = {}
        d = self.root / image_id / "revisions"
        for ann in sorted(p.name for p in d.iterdir()) if d.exists() else []:
            for f in sorted((d / ann).glob("*.json"), reverse=True):
                if json.loads(f.read_text()).get("blinded"):
                    out[ann] = read_mask(f.with_suffix(".png"))
                    break
        return out

    def adjudicate(self, image_id: str, adjudicator: str, decision: np.ndarray, rationale: str) -> Path:
        labels = self.independent_labels(image_id)
        if len(labels) < 2:
            raise ValueError("adjudication requires >= 2 independent annotations")
        dis = disagreement_map(list(labels.values()))
        p = self._next(self.root / image_id / "adjudicated")
        write_png(p, decision)
        meta = {"image_id": image_id, "adjudicator": adjudicator, "rationale": rationale,
                "annotators": sorted(labels), "disagreement_fraction": float((dis > 0).mean()),
                "sha": array_sha256(decision), "inputs_sha": {k: array_sha256(v) for k, v in labels.items()}}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self._log(action="adjudication", path=str(p), **meta)
        return p


def consensus_proposal(labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Predeclared adjudication-assist policy: unanimous pixels keep their label; every other
    pixel becomes UNCERTAIN and is routed to the adjudicator. Returns (proposal, needs_review)."""
    st = np.stack(list(labels.values()))
    unanimous = (st == st[0]).all(0) & (st[0] != UNCERTAIN)
    out = np.where(unanimous, st[0], UNCERTAIN).astype(np.uint8)
    return out, ~unanimous
