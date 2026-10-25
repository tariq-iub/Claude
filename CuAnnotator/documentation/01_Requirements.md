# 01 — Requirements and Traceability

CuAnnotator is a Windows desktop application for building pixel-accurate, scientifically traceable ground-truth
datasets for multiclass copper-corrosion segmentation. The table below maps each principal requirement to the
code that implements it and the test that checks it. "Verified" means an automated test in this repository
exercises the behaviour; "Implemented (UI)" means the feature exists in the WPF front-end, which has not
been run in this build environment (see *Verification status* at the end).

## Functional requirements

| # | Requirement | Implementation | Verification |
|---|---|---|---|
| F1 | Load JPEG/PNG/TIFF/BMP, 8-bit RGB; 16-bit TIFF via documented conversion | `Vision/ImageIO/ImageLoader.cs` | `PersistenceTests`, acceptance test |
| F2 | Record filename, path, size, format, checksum, orientation, ICC presence, specimen, calibration | `ImageLoader`, `ProjectService.ImportImagesAsync`, `ImageItem` | acceptance test (SHA-256, relative path) |
| F3 | Never modify originals | images are copied byte-identically or referenced; all writes go to project/export folders | design + checksum verification on load/export |
| F4 | RGB / HSV / CIELAB / grayscale inspection | `ColorScience`, `EditorViewModel.UpdateCursor` | `ColorScienceTests` |
| F5 | CIEDE2000 tested against published pairs | `ColorScience.DeltaE2000` | 34 Sharma et al. pairs, tolerance 1e-4 |
| F6 | ROI → automatic irregular contour → refinement → acceptance | `AnnotationSession` | `AcceptanceScenarioTests` |
| F7 | GrabCut: rect/mask init, FG/BG seeds, iterative refinement reusing state | `GrabCutSegmenter` | `GrabCut_ForegroundAndBackgroundSeeds_AreHardConstraints` |
| F8 | CIELAB colour-guided segmentation with connectivity and edge constraints | `LabColorSegmenter` | `LabColor_*`, `MinComponentArea_*` |
| F9 | Marker-controlled watershed | `WatershedSegmenter` | `Algorithm_RecoversConcaveIrregularRegion(Watershed)` |
| F10 | Edge-guided active contours | `ActiveContourSegmenter` (MorphACWE, MorphGAC) | `Algorithm_RecoversConcaveIrregularRegion(ActiveContour)` |
| F11 | Hybrid pipeline with explicit stages and stage toggles | `HybridSegmenter` | `Hybrid_ReportsIntermediateStages` |
| F12 | Post-processing (open/close/holes/min area/CC/smoothing/clipping) | `MaskPostProcessor` | `PreserveHoles_*`, `MinComponentArea_*` |
| F13 | Polygons with holes and multiple components; lossless relative to mask | `ContourExtractor`, `MaskRasterizer` | 10 random round-trip tests (exact) |
| F14 | Simplification never alters the authoritative mask | `PolygonSimplification` only affects the vector view | `Simplification_ReducesVerticesWithoutTouchingTheMask` |
| F15 | FG/BG/boundary brushes, vertex editing, union/intersection/subtraction/split/merge/duplicate/delete/relabel | `AnnotationSession`, `ToolController` | `UndoRedo_CoversBrushMergeSplitRelabelDelete`, `Split_*` |
| F16 | Undo/redo for all edits; long-running algorithms cannot corrupt history | `UndoRedoStack`, `ImageSnapshotCommand`, candidate versioning | `StaleSegmentationResult_IsDiscarded…` |
| F17 | Configurable taxonomy with permanent IDs, reserved 0/255 | `Taxonomy`, `TaxonomyValidator`, `TaxonomyEditorWindow` | `TaxonomyTests` |
| F18 | Verification status + evidence separated from class | `VerificationStatus`, `EvidenceReference`, `ClassSuggestion` | `ClassSuggester_*` (status unchanged) |
| F19 | Overlap policies, deterministic | `OverlapPolicy`, `AcceptCandidate`, `LabelMapComposer` (Z-order) | `OverlapPolicies_AreDeterministic` |
| F20 | Semantic 8-bit class-ID PNG at source size; binary masks | `LabelMapComposer`, `MaskIO` | acceptance test steps 25–27, `LabelPng_IsLosslessSingleChannel` |
| F21 | COCO with RLE for holes | `CocoExporter`, `CocoRle` | `Coco_UsesRleForHoles…`, pycocotools string test |
| F22 | YOLO with explicit warnings | `YoloExporter` | `Yolo_NormalisesCoordinates_AndWarns…` |
| F23 | Native lossless JSON with schema version | `NativeJsonSerializer` | `NativeRoundTrip_*`, `FutureSchemaVersion_IsRejected` |
| F24 | SQLite index | `SqliteProjectIndex` | acceptance test (query by class, revision count) |
| F25 | Autosave, atomic writes, backup, recovery | `AtomicFile`, `ProjectStore` | `Save_KeepsBackup…`, `Autosave_RecoveryData…` |
| F26 | Revision history | `AnnotationRevision`, `RestoreRevision` | acceptance test (revision 2, history 1) |
| F27 | QA: coverage, distribution, validation | `QualityAnalyzer` | `QualityAnalyzer_ReportsCoverageAndIssues` |
| F28 | IoU, Dice, precision, recall, boundary F1, κ, pixel agreement | `SegmentationMetrics` | `Metrics_*`, `BoundaryF1_*`, `CohenKappa_*` |
| F29 | Region statistics CSV | `RegionStatistics`, `RegionStatisticsCsv` | used by `ClassSuggester` tests |
| F30 | Class-assisted suggestions, uncalibrated score | `ClassSuggester` | `ClassSuggester_SuggestsSimilarClass…` |
| F31 | Smart region suggestions; batch assistance | `RegionSuggester` | Implemented (UI) |
| F32 | Specimen-grouped, seeded splits | `DatasetSplitter` | `Splitter_IsReproducible_AndNeverLeaksSpecimens` |
| F33 | Audit log separating user and algorithm actions | `JsonlAuditLog`, `AnnotationSession.Log` | acceptance test (actor prefixes) |
| F34 | Coordinate transforms independent of zoom/pan/DPI | `ViewTransform` | `ViewTransformTests` |
| F35 | Tiled/downsampled display for large images | `TiledImageSource` | Implemented (UI) |
| F36 | Keyboard-first, configurable shortcuts | `KeyMapService`, `MainWindow.OnPreviewKeyDown` | Implemented (UI) |

## Non-functional requirements

* **CPU only, offline.** No CUDA, no network access, no model downloads.
* **Responsiveness.** Segmentation runs on the thread pool (`SegmentationAlgorithmBase.SegmentAsync`), is
  cancellable between GrabCut iterations / contour iterations, and stale results are discarded.
* **Reproducibility.** GrabCut seeds OpenCV's RNG (`RandomSeed`); every accepted annotation stores the full
  parameter dictionary, algorithm name and version; splits are seeded.
* **No data loss.** Saves are atomic with a backup; the serializer self-checks the output before replacing
  the project file; autosave writes a separate recovery file.

## Identified technical incompatibilities and resolutions

| Issue | Resolution |
|---|---|
| OpenCV's float `BGR2Lab` approximates the sRGB gamma curve (≈0.2 ΔE76 deviation; measured by `OpenCvFloatLab_DeviatesSlightly…`) | Exact CIELAB is computed in C# with a 256-entry linearisation LUT; OpenCV's conversion is not used for quantitative work. |
| COCO polygons cannot represent holes; pycocotools rasterises polygons with its own rule | Annotations with holes are written as compressed RLE; an `AlwaysRle` mode is pixel-exact. |
| YOLO segmentation cannot represent holes or multipart instances | Explicit policy (skip or outer boundary) with a warning per annotation. |
| WPF cannot be executed on Linux | The UI compiles with `EnableWindowsTargeting`; all behaviour lives in UI-independent services that are tested headlessly. |
| Very large polygons (thousands of exact boundary vertices) are impractical to edit by hand | The vertex editor offers a simplified outline (ε configurable) and states that the mask will be re-rasterised; the exact mask remains in the revision history. |

## Verification status

All automated tests are executed with `dotnet test` (131 tests at the time of writing, all passing on
Linux x64 with the OpenCV 4.13 native runtime). The WPF application compiles with zero warnings but its
interactive behaviour (rendering, mouse and keyboard handling, dialogs) has **not** been exercised in an
automated way; see `10_Testing_and_Validation.md` for the manual UI test checklist.
