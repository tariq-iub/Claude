# CuAnnotator — Copper Corrosion Annotation Studio

An offline, CPU-only Windows desktop application (C# / .NET 10 / WPF / OpenCV) for creating **pixel-accurate,
scientifically traceable ground-truth datasets** for multiclass copper-corrosion segmentation.

Draw a loose rectangle around a corrosion region → an algorithm proposes the irregular boundary → refine it with
foreground/background/boundary strokes → accept with a class → record verification evidence → export
class-ID masks, COCO, YOLO and lossless native JSON.

## Highlights

* **Segmentation engine** (OpenCvSharp, CPU): GrabCut with seeds and state reuse, CIELAB colour-guided region
  growing with **CIEDE2000** (validated against all 34 Sharma et al. reference pairs), marker-controlled
  watershed, morphological active contours (Chan–Vese and geodesic), and a hybrid GrabCut + CIELAB + edge
  pipeline with inspectable intermediate stages. Configurable post-processing (minimum area can be 0 to keep
  microstructures).
* **Pixel accuracy:** the raster mask is authoritative; polygons (with holes and multiple components) convert
  to and from masks **exactly**; one documented coordinate convention, unit-tested view transforms independent
  of zoom, pan and DPI.
* **Scientific integrity:** class, visual suggestion, expert review and instrumental verification are separate
  fields; evidence (XRD, Raman, FTIR, SEM-EDS, XPS …) is recorded; suggestions are uncalibrated similarities,
  never probabilities.
* **Editing:** brushes, vertex editor, split, merge, union/intersection/subtraction, fill, duplicate, relabel,
  re-segment — all undoable, with per-annotation revision history and an audit trail separating user actions
  from algorithm proposals.
* **Persistence:** atomic saves with backup, autosave recovery, versioned native JSON schema, SQLite index.
* **Exports:** 8-bit class-ID PNG masks at source resolution (re-read and validated after export), binary
  masks, COCO (RLE for holes), YOLO (explicit warnings), native JSON, metadata, specimen-grouped seeded splits.
* **Research tools:** QA (coverage, class distribution, validation), IoU/Dice/precision/recall/boundary F1/κ,
  inter-annotator agreement, algorithm comparison, annotation-efficiency logging, region statistics CSV,
  smart region suggestions and batch assistance.

## Quick start

```powershell
# Windows, .NET 10 SDK
dotnet build CuAnnotator.sln -c Release
dotnet test CuAnnotator.sln -c Release
dotnet run --project src/CuAnnotator.Desktop -c Release
# or open CuAnnotator.sln in Visual Studio 2026 and start CuAnnotator.Desktop
```

Release package (self-contained win-x64 + sources + docs + sample + tests + benchmarks):

```powershell
./scripts/publish.ps1          # → artifacts/CuAnnotator-Release.zip
```

Open the example: start the app → **Open** → `samples/SampleProject`.

## Repository layout

| Path | Contents |
|---|---|
| `src/CuAnnotator.Domain` | entities, geometry, raster masks, COCO RLE, view transform, taxonomy |
| `src/CuAnnotator.Vision` | segmentation algorithms, colour science, contours, image I/O |
| `src/CuAnnotator.Application` | annotation session (undo/redo, audit), projects, exports, evaluation |
| `src/CuAnnotator.Infrastructure` | native JSON, atomic project store, SQLite index, audit log, logging |
| `src/CuAnnotator.Desktop` | WPF application |
| `tests/` | xUnit tests incl. the end-to-end acceptance scenario |
| `samples/` | sample generator and generated `SampleProject` |
| `benchmarks/` | performance harness and measured results |
| `documentation/` | 15 technical and user documents |
| `scripts/` | build, test, publish, sample, benchmark scripts |

## Verification status (honest summary)

* **Automated tests:** 131 xUnit tests (domain, vision, export, integration) pass on Linux x64 with the
  OpenCV 4.13 native runtime, including the full section-27 acceptance scenario executed headlessly through
  the same `AnnotationSession` the UI uses (3000×2000 image → ROI → GrabCut → FG/BG corrections → accept as
  Cu₂O/Visually Estimated → save → reopen → polygon edit → revision 2 → export → mask re-read with exact
  pixel/class checks → COCO and native JSON checks → audit log and SQLite checks).
* **Desktop UI:** compiles with zero warnings (`EnableWindowsTargeting`), but this repository's build
  environment is Linux, so the WPF application has **not** been launched or interactively tested here.
  Please run the manual checklist in `documentation/10_Testing_and_Validation.md` on Windows.
* **Benchmarks:** measured numbers are in `benchmarks/results/` with the hardware they were measured on; they
  were not measured on the target i7-10th-gen machine.
* **Release zip:** produced by `scripts/publish.ps1`/`publish.sh`; the committed repository contains sources and
  the sample project, not binaries.

## Documentation

`documentation/01_Requirements.md` … `15_Reproducibility_Guide.md` — requirements traceability, architecture,
database, data model and coordinate conventions, algorithms (with formulas), CIELAB processing, UI
architecture, persistence, export formats, testing, performance, developer guide, user manual, research
methodology and reproducibility.

## Scientific note

CuAnnotator proposes boundaries and records labels; it does not identify chemical phases. Colour-based
proposals and class suggestions are visual aids. Chemical identity should be supported by analytical evidence
recorded in each annotation's verification fields.
