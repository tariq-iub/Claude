# 10 — Testing and Validation

Run everything with `dotnet test CuAnnotator.sln -c Release` (or `scripts/build.ps1` / `build.sh`).
67 test methods expand to **131 test cases** (theories with data rows). All pass on Linux x64 (.NET 10.0,
OpenCV 4.13.0 native runtime) at the time of writing.

## Unit tests

| Project | Covers |
|---|---|
| `CuAnnotator.Domain.Tests` (31 cases) | view transform inverse, zoom-at-cursor anchor, 25 % zoom pixel mapping under pan/zoom, floor rule, fit, DPI-aware 100 %; rectangle operations; shoelace area, perimeter, holes; self-intersection; Douglas–Peucker; raster-mask boolean ops, reframe/trim, clipping, centroid, crack perimeter; COCO RLE round trip, column-major order, **pycocotools reference string**; taxonomy defaults and dataset ID compatibility, validator (duplicates, reserved IDs, colours, shortcuts), edit validation for used IDs; deep clone/restore |
| `CuAnnotator.Vision.Tests` (78 cases) | **CIEDE2000 – all 34 Sharma et al. pairs** + symmetry; sRGB→Lab reference values; exact Lab vs. per-pixel image conversion; OpenCV 8-bit Lab decoding; documented OpenCV float-Lab deviation; HSV; **mask→polygon→mask exactness (10 random masks with holes, islands, single pixels)**; holes and disconnected components; border contact; simplification leaves mask untouched; split/fill; boundary snapping; every algorithm recovers a concave noisy region (IoU > 0.90); GrabCut determinism; FG/BG seeds as hard constraints with state reuse; colour selection vs. connected region growing; hole preservation/filling; min-area 0 keeps 1-px structures; hybrid stage list; out-of-bounds ROI clipping; ROI outside image; empty region; cancellation; parameter round trip |
| `CuAnnotator.Export.Tests` (11 cases) | semantic label map (class IDs, Z-order, ignore, overlap count); lossless single-channel PNG; COCO RLE for holes / polygons otherwise / exact RLE decode / bbox / area / categories; YOLO normalisation and warnings; reproducible specimen-grouped splits (and leakage when grouping is off); full dataset layout with validation; IoU/Dice/precision/recall; boundary F1; Cohen's κ; QA coverage and issues; native JSON RLE storage and corruption detection |

## Integration tests

| Test | Scenario |
|---|---|
| `AcceptanceScenarioTests.MandatoryEndToEndScenario_Cu2O` | Section 27 steps 1–28 on a 3000×2000 synthetic image (see below) |
| `NativeRoundTrip_SemanticMaskIsPixelIdentical_AndAllMetadataSurvives` | save → load → identical label map and identical serialised project; severity, evidence, reviewer, visibility |
| `Save_KeepsBackup_AndCorruptMainFallsBackToBackup` | backup creation; truncated main file → backup loaded with warning |
| `Autosave_RecoveryData_IsDetectedAndDiscardedAfterSave` | recovery file newer than project; discarded after a manual save |
| `FutureSchemaVersion_IsRejected` | schema guard |
| `UndoRedo_CoversBrushMergeSplitRelabelDelete` | 5 edits undone and redone; label map restored; object identity preserved |
| `OverlapPolicies_AreDeterministic` (×3) | PreserveExisting / ReplaceExisting / AllowOverlap results |
| `StaleSegmentationResult_IsDiscarded_WhenCandidateChangesDuringRun` | background result dropped, no phantom history entry |
| `ClassSuggester_SuggestsSimilarClass_WithoutChangingVerification` | k-NN suggestion |

### Acceptance scenario (automated)

1. Create project · 2. import 3000×2000 PNG (checksum, relative path recorded) · 3. decode · 4. loose ROI that
also clips an unwanted similar blob · 5. GrabCut → irregular region (area < 80 % of ROI, > 50 contour
vertices) · 6. foreground stroke on the target, background stroke on the blob · 7. re-run (state reused) →
IoU with ground truth > 0.95, not worse than the first run, blob excluded · 8. accept as Cu₂O (2), Visually
Estimated, with notes · 9. undo/redo acceptance (same object instance) · 10. save · 11. reopen in a fresh store ·
12. locate annotation, mask identical · 13. drag a polygon vertex → revision 2, history 1, area increased ·
14. save · 15. export semantic mask, native JSON, COCO · 16. re-read mask: 3000×2000; **every pixel equals 2
inside the edited mask and 0 elsewhere** · 17. COCO category, area and bbox · 18. native JSON: ID,
verification, notes, revision, mask · 19. audit log: user vs. algorithm actors, corrections, acceptance, class
assignment, polygon edit · 20. SQLite index: class query, revision 2, one revision row.

## UI tests — manual checklist (Windows)

The WPF front-end compiles but could not be executed in the Linux build environment used to develop this
version. Before relying on a build, run this checklist on Windows 10/11 (100 % and 150 % display scaling):

| # | Check | Expected |
|---|---|---|
| U1 | New project, import the two sample PNGs | thumbnails appear; status bar shows size and format |
| U2 | Mouse-wheel zoom over a feature | the feature stays under the cursor |
| U3 | Space+drag, middle-drag, Select-tool drag | image pans; annotations stay registered |
| U4 | Fit (F), 100 % (Ctrl+1) at 150 % Windows scaling | 100 % shows 1 image px per device px (zoom text "100 %") |
| U5 | Zoom ≥ 800 % | pixel grid aligned with pixel edges; contours through pixel centres |
| U6 | R-drag around Cu₂O region | cyan candidate follows the boundary; status shows px/time |
| U7 | N-stroke on the copper island, B-stroke on missed pixels | candidate updates (live preview) |
| U8 | Enter / key 2 | annotation coloured as Cu₂O; flash message; selection shown in panel |
| U9 | Change verification, add evidence, notes, severity | values persist after Save / reopen |
| U10 | Ctrl+Z / Ctrl+Y repeatedly | geometry, class and candidate revert/reapply |
| U11 | Vertex editor: drag, double-click edge, right-click vertex, Ctrl+drag | polygon and fill update; revision increments |
| U12 | Split, merge, union/intersect/subtract, fill holes, duplicate, delete | expected geometry |
| U13 | Magic wand, colour selection, polygon ROI, lasso, smart contour, boundary brush | candidates created |
| U14 | Overlay modes, opacity, minimap navigation, dark theme | correct rendering, readable text |
| U15 | Close with unsaved changes; kill the process after an autosave; reopen | prompt to save; recovery offered |
| U16 | Export dataset; open `masks/*.png` in ImageJ | 8-bit, same size, class values |
| U17 | Open a ≥ 8K image | navigation remains responsive (tiles), segmentation runs in background with progress bar |

## Validation of algorithms on real data

Synthetic tests prove correctness of the implementation, not suitability for a given dataset. Use the
**Evaluate** command (algorithm comparison against reference masks) and the QA report on a held-out sample of
your own images, as described in `14_Research_Methodology.md`.
