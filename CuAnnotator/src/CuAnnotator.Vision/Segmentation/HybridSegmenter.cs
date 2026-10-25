using System.Globalization;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.GrabCut;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using OpenCvSharp;

namespace CuAnnotator.Vision.Segmentation;

/// <summary>
/// Hybrid pipeline with explicit, individually switchable stages:
/// <code>
/// ROI → [GrabCut] → M1 → [CIELAB refinement] → M2 → [geodesic edge refinement] → M3 → post-processing → contours
/// </code>
/// CIELAB refinement: r = median Lab of M1 (robust to boundary contamination);
/// M2 = (M1 ∧ ΔE ≤ T) ∨ (dilate(M1, 2 px) ∧ ΔE ≤ T/2) ∨ seeds — removes colour outliers GrabCut
/// accepted and recovers thin, very similar boundary pixels it missed.
/// Edge refinement: a short MorphGAC run (balloon 0) initialised from M2 snaps the front to gradients.
/// Every intermediate mask is returned in <see cref="SegmentationResult.Stages"/>. The hybrid is not
/// assumed to be better than its constituents; compare them with the evaluation module.
/// </summary>
public sealed class HybridSegmenter : SegmentationAlgorithmBase
{
    public override string Name => "Hybrid GrabCut + CIELAB + Edge";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.Hybrid;

    protected override RasterMask Run(SegmentationContext ctx)
    {
        var frame = ctx.Roi;
        using var pos = ctx.StrokeMask(ctx.Request.PositiveStrokes, frame);
        using var neg = ctx.StrokeMask(ctx.Request.NegativeStrokes, frame);
        var lab = LabImage.FromBgr(ctx.Request.Image, frame);

        // Stage 1 — initial estimate.
        Mat m;
        if (ctx.P.HybridUseGrabCut)
        {
            using var gc = GrabCutSegmenter.RunGrabCut(ctx, out var gcFrame);
            m = MatInterop.MaskToMat(MatInterop.MatToMask(gc, gcFrame), frame);
            ctx.AddStage("1-grabcut", MatInterop.MatToMask(m, frame), "GrabCut foreground");
        }
        else
        {
            var (reference, seed) = LabColorSegmenter.ReferenceColor(ctx, lab, pos);
            seed.Dispose();
            var d0 = lab.DistanceMap(reference, ctx.P.ColorMetric);
            var b0 = new byte[d0.Length];
            for (int i = 0; i < b0.Length; i++) b0[i] = d0[i] <= ctx.P.ColorThreshold ? (byte)1 : (byte)0;
            m = MatInterop.FromBytes(b0, frame.Width, frame.Height);
            ctx.AddStage("1-colour", MatInterop.MatToMask(m, frame), "Colour threshold (GrabCut disabled)");
        }
        using var stage = m;
        ctx.Token.ThrowIfCancellationRequested();

        // Stage 2 — CIELAB refinement.
        if (ctx.P.HybridUseLabRefinement && Cv2.CountNonZero(stage) > 0)
        {
            var sel = MatInterop.ToBytes(stage);
            var reference = lab.Median(sel)!.Value;
            var d = lab.DistanceMap(reference, ctx.P.ColorMetric);
            using var k = MaskPostProcessor.Disk(2);
            using var dil = new Mat();
            Cv2.Dilate(stage, dil, k);
            var dilB = MatInterop.ToBytes(dil);
            var posB = MatInterop.ToBytes(pos);
            double t = ctx.P.ColorThreshold;
            int removed = 0, added = 0;
            for (int i = 0; i < sel.Length; i++)
            {
                byte v;
                if (posB[i] != 0) v = 1;
                else if (sel[i] != 0) v = d[i] <= t ? (byte)1 : (byte)0;
                else v = dilB[i] != 0 && d[i] <= t / 2 ? (byte)1 : (byte)0;
                if (sel[i] != 0 && v == 0) removed++;
                if (sel[i] == 0 && v != 0) added++;
                sel[i] = v;
            }
            System.Runtime.InteropServices.Marshal.Copy(sel, 0, stage.Data, sel.Length);
            LabColorSegmenter.ApplyBarriers(ctx, stage, frame);
            ctx.Diagnostics["hybrid.lab.reference"] = reference.ToString();
            ctx.Diagnostics["hybrid.lab.removed"] = removed.ToString(CultureInfo.InvariantCulture);
            ctx.Diagnostics["hybrid.lab.added"] = added.ToString(CultureInfo.InvariantCulture);
            ctx.AddStage("2-cielab", MatInterop.MatToMask(stage, frame), $"CIELAB refinement (T = {t})");
        }
        ctx.Token.ThrowIfCancellationRequested();

        // Stage 3 — edge refinement.
        Mat refined = stage;
        Mat? owned = null;
        if (ctx.P.HybridUseEdgeRefinement && Cv2.CountNonZero(stage) > 0)
        {
            owned = ActiveContourSegmenter.Evolve(ctx, frame, stage, ActiveContourMode.GeodesicEdge,
                ctx.P.HybridEdgeIterations, pos, neg);
            refined = owned;
            ctx.AddStage("3-edge", MatInterop.MatToMask(refined, frame), "Geodesic edge refinement");
        }

        // Stage 4 — post-processing and connected-component analysis.
        foreach (var kv in MaskPostProcessor.Apply(refined, PostProcessingOptions.From(ctx.P), null, pos, neg))
            ctx.Diagnostics[kv.Key] = kv.Value;
        var result = MatInterop.MatToMask(refined, frame);
        ctx.AddStage("4-postprocessed", result, "Morphological cleanup and component filtering");
        owned?.Dispose();
        return result;
    }
}
