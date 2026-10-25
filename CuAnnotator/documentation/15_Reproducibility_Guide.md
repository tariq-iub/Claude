# 15 — Reproducibility Guide

## What is recorded

| Item | Where |
|---|---|
| Application version | project file (`appVersion`, `createdWithAppVersion`, `savedWithAppVersion`), dataset metadata |
| Project schema version | `schemaVersion` in every native file |
| Algorithm name and implementation version | `Annotation.SegmentationAlgorithm`, `AlgorithmVersion`, audit actor `algorithm:Name@x.y.z` |
| All processing parameters (incl. random seed, quality preset, ROI) | `Annotation.AlgorithmParameters` (invariant-culture strings) |
| Original image checksum (SHA-256), size, format, orientation, ICC presence | `ImageItem`, `metadata/dataset.json` |
| Taxonomy and its version | project file, every per-image JSON, `metadata/classes.json` |
| Timestamps (ISO 8601 UTC) | annotations, revisions, audit log |
| Verification status and evidence | annotations, `metadata/verification.json` |
| User corrections vs. algorithm proposals | audit log actors (`user:` / `algorithm:`) and actions (Foreground/Background/Barrier Correction, Segmentation, Accepted, Polygon Edited …) |
| Revisions | `RevisionHistory` (with masks), SQLite `revisions` |
| Dataset split seed, fractions, grouping | `metadata/dataset.json` |

Example audit trail (Audit button):

```
2026-10-25 10:31:12 UTC
Image: specimen_001.png
Action: ROI Created  [user:researcher]
Coordinates: (1150, 720, 1640, 1200)

2026-10-25 10:31:15 UTC
Action: GrabCut Segmentation  [algorithm:GrabCut@1.0.0]
Candidate boundary: 101233 px, 1 component(s), 812 ms; Algorithm=GrabCut, GrabCutIterations=5, …

2026-10-25 10:31:21 UTC
Action: Foreground Correction  [user:researcher]

2026-10-25 10:31:27 UTC
Action: Segmentation Accepted  [user:researcher]

2026-10-25 10:31:30 UTC
Action: Class Assigned  [user:researcher]
Class: Cu2O (2); Verification: VisuallyEstimated
```

## Determinism

* GrabCut: OpenCV's RNG is seeded with `RandomSeed` before every run (test: identical masks on repeated runs).
* All other algorithms are deterministic given identical inputs and parameters.
* Thread-parallel loops (Lab conversion, ΔE maps) write disjoint outputs; results do not depend on scheduling.
* Region suggestions downsample deterministically (INTER_AREA).
* Results can differ across OpenCV versions; the OpenCV version is pinned in `Directory.Build.props` and printed
  in benchmark reports.

## Reproducing a segmentation

1. Load the same image (verify SHA-256).
2. Use the stored `roi`, `Algorithm` and parameter dictionary (`SegmentationParameters.FromDictionary`).
3. Re-apply the guidance strokes — these are recorded in the audit log as counts, and the accepted mask itself is
   stored exactly, so the *annotation* is always reproducible even if the interactive path is not replayed.

## Reproducing a dataset

Re-export from the saved project with the same seed and options; masks are bit-identical (export validation
re-reads every mask and compares it with the project).

## Recommended reporting checklist

* CuAnnotator version, OpenCV version, OS.
* Taxonomy version and class definitions; verification policy (which levels were included).
* Algorithms and parameter ranges used; whether hybrid stages were enabled.
* Split seed, grouping, fractions.
* QA summary: coverage, class distribution, overlap handling, inter-annotator agreement (with BF tolerance).
