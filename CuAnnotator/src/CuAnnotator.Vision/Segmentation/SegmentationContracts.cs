using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using OpenCvSharp;

namespace CuAnnotator.Vision.Segmentation;

/// <summary>User guidance: a polyline in image coordinates painted with a brush of the given radius.</summary>
public sealed record Stroke(IReadOnlyList<PixelPoint> Points, int Radius)
{
    public static Stroke Point(PixelPoint p, int radius = 0) => new(new[] { p }, radius);
}

public enum ProcessingQuality { Fast, Balanced, Precise }
public enum ActiveContourMode { RegionChanVese, GeodesicEdge }

/// <summary>
/// All tunable parameters of the segmentation engine. Values are configurable defaults, not
/// universally optimal settings; the exact values used are stored with every accepted annotation.
/// </summary>
public sealed class SegmentationParameters
{
    public SegmentationAlgorithmKind Algorithm { get; set; } = SegmentationAlgorithmKind.GrabCut;
    public ProcessingQuality Quality { get; set; } = ProcessingQuality.Balanced;

    // GrabCut
    public int GrabCutIterations { get; set; } = 5;
    /// <summary>Context added around the ROI (fraction of ROI size) so the background model sees real background.</summary>
    public double RoiContextMargin { get; set; } = 0.15;
    /// <summary>Seed of OpenCV's RNG used by GrabCut's GMM initialisation (makes runs deterministic).</summary>
    public ulong RandomSeed { get; set; } = 12345;

    // CIELAB colour guidance
    public ColorDifferenceMetric ColorMetric { get; set; } = ColorDifferenceMetric.CIEDE2000;
    public double ColorThreshold { get; set; } = 12.0;
    public bool RequireConnectivity { get; set; } = true;
    public bool UseEdgeBarrier { get; set; }
    public double CannyLow { get; set; } = 40;
    public double CannyHigh { get; set; } = 120;

    // Watershed
    public double WatershedPreBlurSigma { get; set; } = 1.0;
    public double WatershedAutoMarkerDeltaE { get; set; } = 6.0;

    // Active contours (morphological snakes)
    public ActiveContourMode ActiveContourMode { get; set; } = ActiveContourMode.RegionChanVese;
    public int ActiveContourIterations { get; set; } = 150;
    public int ActiveContourSmoothing { get; set; } = 1;
    /// <summary>Region influence: weight of the inside term (λ1) in Chan–Vese.</summary>
    public double RegionLambdaInside { get; set; } = 1.0;
    /// <summary>Weight of the outside term (λ2) in Chan–Vese.</summary>
    public double RegionLambdaOutside { get; set; } = 1.0;
    /// <summary>Edge attraction strength α in g(I) = 1/sqrt(1 + α|∇Gσ*I|).</summary>
    public double EdgeAttraction { get; set; } = 100;
    public double EdgeSigma { get; set; } = 1.5;
    /// <summary>Balloon force for the geodesic mode (&gt;0 inflate, &lt;0 shrink).</summary>
    public double Balloon { get; set; }
    /// <summary>Stop when fewer than this fraction of ROI pixels change in an iteration.</summary>
    public double ConvergenceTolerance { get; set; } = 0.0005;

    // Hybrid stages
    public bool HybridUseGrabCut { get; set; } = true;
    public bool HybridUseLabRefinement { get; set; } = true;
    public bool HybridUseEdgeRefinement { get; set; } = true;
    public int HybridEdgeIterations { get; set; } = 25;

    // Post-processing
    public bool MorphologicalCleanup { get; set; } = true;
    public int OpeningRadius { get; set; } = 1;
    public int ClosingRadius { get; set; } = 1;
    public bool PreserveHoles { get; set; } = true;
    /// <summary>Components smaller than this are removed; 0 disables the filter.</summary>
    public int MinComponentArea { get; set; } = 5;
    public bool KeepOnlySeededComponents { get; set; }
    /// <summary>Gaussian σ (pixels) for boundary smoothing of the mask; 0 disables.</summary>
    public double BoundarySmoothing { get; set; }
    /// <summary>Douglas–Peucker tolerance for the editable polygon view only (mask unaffected).</summary>
    public double PolygonSimplification { get; set; }

    public SegmentationParameters Clone() => (SegmentationParameters)MemberwiseClone();

    private static readonly PropertyInfo[] Props = typeof(SegmentationParameters).GetProperties();

    /// <summary>Invariant-culture string dictionary used for persistence and audit trails.</summary>
    public Dictionary<string, string> ToDictionary() =>
        Props.ToDictionary(p => p.Name, p => Convert.ToString(p.GetValue(this), CultureInfo.InvariantCulture) ?? "");

    public static SegmentationParameters FromDictionary(IReadOnlyDictionary<string, string> values)
    {
        var result = new SegmentationParameters();
        foreach (var p in Props)
        {
            if (!values.TryGetValue(p.Name, out var s)) continue;
            try
            {
                object v = p.PropertyType.IsEnum
                    ? Enum.Parse(p.PropertyType, s)
                    : Convert.ChangeType(s, p.PropertyType, CultureInfo.InvariantCulture);
                p.SetValue(result, v);
            }
            catch (Exception ex) when (ex is FormatException or ArgumentException or OverflowException) { /* keep default */ }
        }
        return result;
    }

    /// <summary>
    /// Maximum context-frame pixel count GrabCut processes at full resolution for the quality preset. Larger
    /// frames are downscaled. Budgets were chosen from measured GrabCut cost, which grows faster than linearly
    /// with frame size (see benchmarks/results and 11_Performance_Optimization.md).
    /// </summary>
    public long MaxProcessingPixels => Quality switch
    {
        ProcessingQuality.Fast => 512 * 512,
        ProcessingQuality.Balanced => 768 * 768,
        _ => long.MaxValue,
    };
}

/// <summary>Opaque GrabCut state (4-state mask and GMMs) so refinement continues from the previous run.</summary>
public sealed class GrabCutState
{
    public required PixelRect Frame { get; init; }
    public required PixelRect Roi { get; init; }
    public required byte[] StateMask { get; init; }
    public required double[] BackgroundModel { get; init; }
    public required double[] ForegroundModel { get; init; }
    public required double DownscaleFactor { get; init; }
}

public sealed class SegmentationRequest
{
    /// <summary>Full source image, BGR 8-bit 3-channel. Not owned or modified by the algorithm.</summary>
    public required Mat Image { get; init; }
    public required PixelRect Roi { get; init; }
    public RasterMask? InitialMask { get; init; }
    public IReadOnlyList<Stroke> PositiveStrokes { get; init; } = Array.Empty<Stroke>();
    public IReadOnlyList<Stroke> NegativeStrokes { get; init; } = Array.Empty<Stroke>();
    /// <summary>Boundary constraints (e.g. snapped boundary-brush paths) the region may not cross.</summary>
    public IReadOnlyList<Stroke> BarrierStrokes { get; init; } = Array.Empty<Stroke>();
    public GrabCutState? PreviousGrabCutState { get; init; }
    public SegmentationParameters Parameters { get; init; } = new();
}

public sealed record IntermediateStage(string Name, RasterMask Mask, string Description);

public sealed class SegmentationResult
{
    public required string AlgorithmName { get; init; }
    public required string AlgorithmVersion { get; init; }
    public required RasterMask Mask { get; init; }
    public required IReadOnlyList<PolygonComponent> Components { get; init; }
    public required TimeSpan ProcessingTime { get; init; }
    public required Dictionary<string, string> ParametersUsed { get; init; }
    public Dictionary<string, string> Diagnostics { get; init; } = new();
    public List<IntermediateStage> Stages { get; init; } = new();
    public List<string> Warnings { get; init; } = new();
    public GrabCutState? GrabCutState { get; init; }
}

public interface ISegmentationAlgorithm
{
    string Name { get; }
    string Version { get; }
    SegmentationAlgorithmKind Kind { get; }
    Task<SegmentationResult> SegmentAsync(SegmentationRequest request, CancellationToken cancellationToken);
}

/// <summary>Shared plumbing: validation, background execution, timing and contour extraction.</summary>
public abstract class SegmentationAlgorithmBase : ISegmentationAlgorithm
{
    public abstract string Name { get; }
    public abstract string Version { get; }
    public abstract SegmentationAlgorithmKind Kind { get; }

    public Task<SegmentationResult> SegmentAsync(SegmentationRequest request, CancellationToken cancellationToken)
    {
        Validate(request);
        // Always off the caller's (UI) thread.
        return Task.Run(() =>
        {
            var sw = Stopwatch.StartNew();
            var ctx = new SegmentationContext(request, cancellationToken);
            var mask = Run(ctx);
            cancellationToken.ThrowIfCancellationRequested();
            var components = Contours.ContourExtractor.Extract(mask, request.Parameters.PolygonSimplification);
            sw.Stop();
            var pars = request.Parameters.ToDictionary();
            pars["Algorithm"] = Kind.ToString();
            return new SegmentationResult
            {
                AlgorithmName = Name,
                AlgorithmVersion = Version,
                Mask = mask,
                Components = components,
                ProcessingTime = sw.Elapsed,
                ParametersUsed = pars,
                Diagnostics = ctx.Diagnostics,
                Stages = ctx.Stages,
                Warnings = ctx.Warnings,
                GrabCutState = ctx.GrabCutState,
            };
        }, cancellationToken);
    }

    protected abstract RasterMask Run(SegmentationContext ctx);

    public static void Validate(SegmentationRequest r)
    {
        if (r.Image is null || r.Image.Empty()) throw new ArgumentException("Source image is empty.");
        if (r.Image.Type() != MatType.CV_8UC3) throw new ArgumentException("Source image must be BGR CV_8UC3.");
        var clipped = r.Roi.ClampTo(r.Image.Width, r.Image.Height);
        if (clipped.IsEmpty) throw new SegmentationException("The ROI lies outside the image.");
        if (clipped.Width < 3 || clipped.Height < 3) throw new SegmentationException("The ROI is too small (minimum 3×3 pixels inside the image).");
    }
}

public sealed class SegmentationException(string message) : Exception(message);

/// <summary>Per-run working state shared by algorithm stages.</summary>
public sealed class SegmentationContext(SegmentationRequest request, CancellationToken token)
{
    public SegmentationRequest Request { get; } = request;
    public CancellationToken Token { get; } = token;
    public SegmentationParameters P => Request.Parameters;
    public PixelRect Roi { get; } = request.Roi.ClampTo(request.Image.Width, request.Image.Height);
    public Dictionary<string, string> Diagnostics { get; } = new();
    public List<IntermediateStage> Stages { get; } = new();
    public List<string> Warnings { get; } = new();
    public GrabCutState? GrabCutState { get; set; }

    public PixelRect ContextFrame(double marginFraction)
    {
        int m = (int)Math.Ceiling(Math.Max(Roi.Width, Roi.Height) * marginFraction);
        return Roi.Inflate(Math.Max(m, 0)).ClampTo(Request.Image.Width, Request.Image.Height);
    }

    /// <summary>0/1 Mat over <paramref name="frame"/> with the given strokes painted.</summary>
    public Mat StrokeMask(IEnumerable<Stroke> strokes, PixelRect frame)
    {
        var m = new Mat(frame.Height, frame.Width, MatType.CV_8UC1, Scalar.All(0));
        Interop.MatInterop.PaintStrokes(m, strokes, frame, 1);
        return m;
    }

    public void AddStage(string name, RasterMask mask, string description)
    {
        Stages.Add(new IntermediateStage(name, mask, description));
        Diagnostics[$"stage.{name}.pixels"] = mask.PixelCount.ToString(CultureInfo.InvariantCulture);
    }
}
