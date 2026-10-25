using System.Globalization;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Vision.ColorAnalysis;

/// <summary>
/// CIELAB colour-guided segmentation.
/// <para>
/// 1. A reference colour r is the per-channel median Lab of the foreground strokes, else of the initial
///    mask, else of the central 20 % of the ROI (recorded in diagnostics).<br/>
/// 2. Candidate pixels satisfy ΔE(p, r) ≤ T (CIEDE2000 by default).<br/>
/// 3. Optional edge barrier: Canny edges (dilated by 1 px) and barrier strokes are removed from the
///    candidate set so regions cannot leak across visible boundaries.<br/>
/// 4. Optional spatial constraint: only 8-connected components touching the seeds are kept
///    (region growing); without it the result is a global colour selection inside the ROI.<br/>
/// 5. Standard post-processing.
/// </para>
/// Colour similarity is not evidence of chemical identity; it only proposes a region.
/// </summary>
public sealed class LabColorSegmenter : SegmentationAlgorithmBase
{
    public override string Name => "CIELAB Colour-Guided";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.LabColor;

    protected override RasterMask Run(SegmentationContext ctx)
    {
        var frame = ctx.Roi;
        var lab = LabImage.FromBgr(ctx.Request.Image, frame);
        ctx.Token.ThrowIfCancellationRequested();
        using var pos = ctx.StrokeMask(ctx.Request.PositiveStrokes, frame);
        using var neg = ctx.StrokeMask(ctx.Request.NegativeStrokes, frame);

        var (reference, seed) = ReferenceColor(ctx, lab, pos);
        using var seedMat = seed;
        var dist = lab.DistanceMap(reference, ctx.P.ColorMetric);
        ctx.Token.ThrowIfCancellationRequested();

        var cand = new byte[dist.Length];
        for (int i = 0; i < cand.Length; i++) cand[i] = dist[i] <= ctx.P.ColorThreshold ? (byte)1 : (byte)0;
        using var m = MatInterop.FromBytes(cand, frame.Width, frame.Height);
        ctx.AddStage("colour-threshold", MatInterop.MatToMask(m, frame), $"ΔE ≤ {ctx.P.ColorThreshold} to reference {reference}");

        ApplyBarriers(ctx, m, frame);
        if (ctx.P.RequireConnectivity)
        {
            using var seeds = seedMat.Clone();
            Cv2.BitwiseAnd(seeds, m, seeds);
            MaskPostProcessor.KeepComponentsTouching(m, seeds, fallbackToLargest: true);
            ctx.AddStage("connected", MatInterop.MatToMask(m, frame), "Components connected to the seed region");
        }

        foreach (var kv in MaskPostProcessor.Apply(m, PostProcessingOptions.From(ctx.P), null, pos, neg))
            ctx.Diagnostics[kv.Key] = kv.Value;
        ctx.Diagnostics["lab.reference"] = reference.ToString();
        ctx.Diagnostics["lab.metric"] = ctx.P.ColorMetric.ToString();
        return MatInterop.MatToMask(m, frame);
    }

    /// <summary>Returns the reference colour and a 0/1 seed Mat (frame-local) describing where it was sampled.</summary>
    internal static (LabColor Reference, Mat Seed) ReferenceColor(SegmentationContext ctx, LabImage lab, Mat positive)
    {
        var frame = lab.Frame;
        Mat seed = positive.Clone();
        string source = "foreground strokes";
        if (Cv2.CountNonZero(seed) == 0 && ctx.Request.InitialMask is { } init && !init.IsEmpty)
        {
            seed.Dispose();
            seed = MatInterop.MaskToMat(init, frame);
            source = "initial mask";
        }
        if (Cv2.CountNonZero(seed) == 0)
        {
            int cw = Math.Max(1, frame.Width / 5), ch = Math.Max(1, frame.Height / 5);
            seed[new Rect((frame.Width - cw) / 2, (frame.Height - ch) / 2, cw, ch)].SetTo(Scalar.All(1));
            source = "ROI centre (20 %)";
        }
        var reference = lab.Median(MatInterop.ToBytes(seed))
                        ?? throw new SegmentationException("No pixels available to sample a reference colour.");
        ctx.Diagnostics["lab.referenceSource"] = source;
        return (reference, seed);
    }

    internal static void ApplyBarriers(SegmentationContext ctx, Mat m, PixelRect frame)
    {
        if (ctx.P.UseEdgeBarrier)
        {
            using var crop = new Mat(ctx.Request.Image, frame.ToRect());
            using var gray = new Mat();
            Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.GaussianBlur(gray, gray, new Size(3, 3), 0);
            using var edges = new Mat();
            Cv2.Canny(gray, edges, ctx.P.CannyLow, ctx.P.CannyHigh);
            m.SetTo(Scalar.All(0), edges);
            ctx.Diagnostics["lab.edgePixels"] = Cv2.CountNonZero(edges).ToString(CultureInfo.InvariantCulture);
        }
        if (ctx.Request.BarrierStrokes.Count > 0)
        {
            using var bar = ctx.StrokeMask(ctx.Request.BarrierStrokes, frame);
            m.SetTo(Scalar.All(0), bar);
        }
    }
}
