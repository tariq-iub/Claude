# 02 — Software Architecture

## Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│ CuAnnotator.Desktop (WPF, net10.0-windows)                               │
│   Views (XAML) · ViewModels (CommunityToolkit.Mvvm) · Controls           │
│   (ImageViewport, Minimap) · Interaction (ToolController) · Rendering    │
│   (TiledImageSource, OverlayFactory) · Services (dialogs, theme, keys)   │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ calls
┌───────────────▼──────────────────────────────────────────────────────────┐
│ CuAnnotator.Application (net10.0)                                        │
│   Annotations: AnnotationSession (editing engine), geometry service,     │
│                ClassSuggester, RegionSuggester, statistics CSV            │
│   Commands: UndoRedoStack, ImageSnapshotCommand                          │
│   Projects: ProjectService      Export: label maps, COCO, YOLO, dataset  │
│   Evaluation: metrics, QA, algorithm comparison, efficiency             │
│   Services: IProjectStore, IProjectSerializer (ports)                    │
└───────┬───────────────────────────────┬──────────────────────────────────┘
        │                               │ implemented by
┌───────▼───────────────┐   ┌───────────▼──────────────────────────────────┐
│ CuAnnotator.Vision    │   │ CuAnnotator.Infrastructure                   │
│ (OpenCvSharp)         │   │   Serialization (native JSON v1)             │
│  GrabCut, CIELAB,     │   │   Persistence (ProjectStore, audit JSONL)    │
│  Watershed, active    │   │   FileStorage (atomic writes)                │
│  contours, hybrid,    │   │   SQLite (project index)                     │
│  morphology, contours │   │   Logging (Serilog)                          │
└───────┬───────────────┘   └───────────┬──────────────────────────────────┘
        │                               │
┌───────▼───────────────────────────────▼──────────────────────────────────┐
│ CuAnnotator.Domain (no dependencies): entities, geometry, RasterMask,    │
│ CocoRle, ViewTransform, taxonomy & validation, enums                     │
│ CuAnnotator.Contracts: COCO DTOs, native format constants                │
└──────────────────────────────────────────────────────────────────────────┘
```

Dependency rule: arrows point inwards. Domain depends on nothing; Vision depends on Domain and OpenCvSharp;
Application depends on Domain, Vision and Contracts and defines the persistence ports that Infrastructure
implements. Application deliberately depends on Vision directly (rather than through an abstraction) because
the segmentation engine *is* the core use case; algorithms are nevertheless pluggable through
`ISegmentationAlgorithm` and `SegmentationEngine`.

## Key design decisions

1. **UI-independent editing engine.** `AnnotationSession` implements the whole annotation workflow (ROI,
   guidance, segmentation, acceptance, edits, undo/redo, audit). The WPF layer only translates input into
   calls; the mandatory acceptance scenario is therefore executed headlessly as an integration test.
2. **Mask is authoritative.** Every annotation stores a `RasterMask` (bounds + 0/1 bytes). Polygons are a
   derived, editable view; polygon edits regenerate the mask by a documented rule (see 04).
3. **Immutable masks, snapshot undo.** Masks are never mutated in place, so `ImageSnapshotCommand` can capture
   the annotation list cheaply (shared mask references) and restore it while preserving object identity.
4. **Candidate versioning.** Each candidate change increments a version; a segmentation result is applied only
   if the version is unchanged when it completes. Stale results are dropped, so background work cannot corrupt
   history.
5. **Exact colour science in managed code.** CIELAB and CIEDE2000 are implemented in C# and validated against
   reference data; OpenCV is used for the heavy image operations (GrabCut, morphology, contours, watershed).
6. **Display pyramid separate from data.** `TiledImageSource` creates downsampled tiles for drawing only;
   segmentation and export always use the full-resolution `Mat`.

## Dependency injection

`App.OnStartup` builds a `ServiceCollection` with singletons for the logger, user settings, serializer,
project store/service, segmentation engine, dataset exporter, theme and dialog services, `MainViewModel` and
`MainWindow`. Per-image objects (`AnnotationSession`, `EditorViewModel`) are created by `MainViewModel` when
an image is opened, because they own native image memory.

## Threading model

* UI thread: all model mutation (`AnnotationSession` is single-threaded by contract).
* Thread pool: decoding, segmentation, region suggestions, export, thumbnails.
* Native memory: the full-resolution `Mat` is owned by `EditorViewModel`; disposal is deferred until all
  background tasks that read it complete (`EditorViewModel.Track`).

## Extensibility

* New algorithm: implement `SegmentationAlgorithmBase.Run(SegmentationContext)` (or `ISegmentationAlgorithm`)
  and register it in `SegmentationEngine.DefaultAlgorithms()`. An ONNX/SAM-style model can be added the same
  way (prompt = ROI + strokes, output = mask) without changing the session or exports.
* New export format: consume `LabelMapComposer` / annotations in a new class under `Application/Export`.
* New metrics: `Application/Evaluation`.
