# 12 — Developer Guide

## Prerequisites

* .NET SDK 10 (Visual Studio 2026 with the ".NET desktop development" workload on Windows).
* Windows 10/11 x64 to run the desktop application. Libraries and tests also build and run on Linux x64.
* No GPU, CUDA or internet connection is needed at run time (NuGet restore needs network access once).

## Build, test, run

```powershell
dotnet restore CuAnnotator.sln
dotnet build CuAnnotator.sln -c Release
dotnet test CuAnnotator.sln -c Release
dotnet run --project src/CuAnnotator.Desktop          # Windows only
```

Scripts: `scripts/build.ps1` / `scripts/build.sh` (restore, build, test), `scripts/publish.ps1` (self-contained
win-x64 publish + release zip), `scripts/generate-sample.*` (sample project), `scripts/benchmark.*`.

## Solution layout

```
src/CuAnnotator.Domain          entities, geometry, RasterMask, CocoRle, ViewTransform, taxonomy (no deps)
src/CuAnnotator.Contracts       COCO DTOs, native-format constants
src/CuAnnotator.Vision          OpenCvSharp algorithms, colour science, contours, image I/O
src/CuAnnotator.Application     AnnotationSession, undo/redo, project service, exports, evaluation
src/CuAnnotator.Infrastructure  JSON serializer, project store, SQLite index, audit log, logging
src/CuAnnotator.Desktop         WPF application (MVVM)
tests/*                         xUnit: Domain, Vision, Export, Integration (acceptance scenario)
benchmarks/                     performance harness (Stopwatch medians, peak memory)
samples/                        sample generator + generated SampleProject
documentation/                  this documentation
```

## Native OpenCV runtime

`OpenCvSharp4` (managed) is referenced by Vision. Executables add the native runtime:
`OpenCvSharp4.runtime.win` (Windows) and `OpenCvSharp4.official.runtime.linux-x64` (tests/tools on Linux).
The version is centralised in `Directory.Build.props` (`OpenCvSharpVersion`).

## Adding a segmentation algorithm

```csharp
public sealed class MyAlgorithm : SegmentationAlgorithmBase
{
    public override string Name => "My algorithm";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.Manual; // add a new enum value

    protected override RasterMask Run(SegmentationContext ctx)
    {
        // ctx.Request.Image (BGR 8UC3, full image), ctx.Roi (clipped), ctx.P (parameters),
        // ctx.StrokeMask(...), ctx.Token (check for cancellation), ctx.AddStage(...), ctx.Diagnostics
        using var m = /* compute a 0/1 Mat over ctx.Roi */;
        MaskPostProcessor.Apply(m, PostProcessingOptions.From(ctx.P), null, pos, neg);
        return MatInterop.MatToMask(m, ctx.Roi);
    }
}
```

Register it in `SegmentationEngine.DefaultAlgorithms()` and add it to `MainViewModel.Algorithms`. Contours,
timing, parameter recording, background execution and audit logging are handled by the base class and session.

## Conventions

* All model mutations go through `AnnotationSession` (undoable, audited). Never mutate `Annotation.Mask` in
  place: create a new `RasterMask`.
* Use `MatInterop` for Mat ↔ byte[]/RasterMask conversions; dispose Mats with `using`.
* Algorithms must be deterministic for fixed parameters (seed any randomness).
* Coordinates: see 04. Never store view coordinates.
* Tests: synthetic images with known ground truth (`tests/CuAnnotator.Vision.Tests/SyntheticImages.cs`).

## Logging

Serilog writes rolling daily logs to `%LOCALAPPDATA%/CuAnnotator/logs` (14 days). Unhandled exceptions are
logged and shown without terminating the application.
