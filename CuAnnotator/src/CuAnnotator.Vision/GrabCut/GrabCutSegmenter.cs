using System.Globalization;
using System.Runtime.InteropServices;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Vision.GrabCut;

/// <summary>
/// OpenCV GrabCut (Rother et al., 2004) with rectangle, mask and seed initialisation.
/// <para>
/// The ROI is processed inside a context frame (ROI + margin) so the background GMM learns from real
/// surroundings. Pixels outside the ROI are definite background (GC_BGD); inside the ROI they start as
/// probable foreground (GC_PR_FGD) or, when an initial mask is given, PR_FGD/PR_BGD from that mask.
/// Foreground strokes become GC_FGD and background strokes GC_BGD. When a previous state for the same
/// ROI exists, its 4-state mask and GMMs are reused and only new seeds are written, so iterative
/// refinement converges from the earlier result instead of restarting.
/// </para>
/// <para>The rectangle is initialisation guidance, not proof that the object was identified.</para>
/// </summary>
public sealed class GrabCutSegmenter : SegmentationAlgorithmBase
{
    public override string Name => "GrabCut";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.GrabCut;

    private const byte BGD = 0, FGD = 1, PR_BGD = 2, PR_FGD = 3;

    protected override RasterMask Run(SegmentationContext ctx)
    {
        var mask = RunGrabCut(ctx, out var frame);
        using var m = mask;
        using var allowed = RoiMask(ctx, frame);
        using var pos = ctx.StrokeMask(ctx.Request.PositiveStrokes, frame);
        using var neg = ctx.StrokeMask(ctx.Request.NegativeStrokes, frame);
        ctx.AddStage("grabcut", MatInterop.MatToMask(m, frame), "Raw GrabCut foreground (FGD ∪ PR_FGD)");
        foreach (var kv in MaskPostProcessor.Apply(m, PostProcessingOptions.From(ctx.P), allowed, pos, neg))
            ctx.Diagnostics[kv.Key] = kv.Value;
        var result = MatInterop.MatToMask(m, frame);
        ctx.AddStage("postprocessed", result, "After morphological post-processing");
        return result;
    }

    internal static Mat RoiMask(SegmentationContext ctx, PixelRect frame)
    {
        var allowed = new Mat(frame.Height, frame.Width, MatType.CV_8UC1, Scalar.All(0));
        var r = ctx.Roi;
        allowed[new Rect(r.X - frame.X, r.Y - frame.Y, r.Width, r.Height)].SetTo(Scalar.All(1));
        return allowed;
    }

    /// <summary>Runs GrabCut and returns a 0/1 foreground Mat over the returned context frame (caller disposes).</summary>
    internal static Mat RunGrabCut(SegmentationContext ctx, out PixelRect frame)
    {
        var req = ctx.Request;
        var prev = req.PreviousGrabCutState;
        bool reuse = prev is not null && prev.Roi == ctx.Roi && req.InitialMask is null;
        frame = reuse ? prev!.Frame : ctx.ContextFrame(ctx.P.RoiContextMargin);
        var f = frame;

        // Optional downscaling for the Fast/Balanced presets on very large ROIs.
        double scale = reuse ? prev!.DownscaleFactor : 1.0;
        if (!reuse && (long)f.Width * f.Height > ctx.P.MaxProcessingPixels)
            scale = Math.Sqrt((double)ctx.P.MaxProcessingPixels / ((long)f.Width * f.Height));
        int w = Math.Max(3, (int)Math.Round(f.Width * scale)), h = Math.Max(3, (int)Math.Round(f.Height * scale));

        using var crop = new Mat(req.Image, f.ToRect());
        using var img = new Mat();
        if (scale < 1) Cv2.Resize(crop, img, new Size(w, h), 0, 0, InterpolationFlags.Area); else crop.CopyTo(img);

        // Full-resolution state, then resized for processing.
        var state = new byte[f.Width * f.Height];
        var roiLocal = new Rect(ctx.Roi.X - f.X, ctx.Roi.Y - f.Y, ctx.Roi.Width, ctx.Roi.Height);
        using var bgdModel = new Mat(); using var fgdModel = new Mat();

        if (reuse && scale == 1.0)
        {
            Array.Copy(prev!.StateMask, state, state.Length);
        }
        else
        {
            for (int y = 0; y < f.Height; y++)
                for (int x = 0; x < f.Width; x++)
                    state[y * f.Width + x] = roiLocal.Contains(x, y) ? PR_FGD : BGD;
            if (req.InitialMask is not null)
            {
                var init = req.InitialMask.Reframe(f).Data;
                for (int i = 0; i < state.Length; i++)
                    if (state[i] == PR_FGD && init[i] == 0) state[i] = PR_BGD;
            }
            if (reuse) // downscaled reuse: rebuild from previous foreground decisions
                for (int i = 0; i < state.Length; i++)
                    if (state[i] != BGD) state[i] = prev!.StateMask[i] is FGD or PR_FGD ? PR_FGD : PR_BGD;
        }

        // Hard constraints from user strokes.
        using (var pos = ctx.StrokeMask(req.PositiveStrokes, f))
        using (var neg = ctx.StrokeMask(req.NegativeStrokes, f))
        using (var bar = ctx.StrokeMask(req.BarrierStrokes, f))
        {
            var p = MatInterop.ToBytes(pos); var n = MatInterop.ToBytes(neg); var b = MatInterop.ToBytes(bar);
            for (int i = 0; i < state.Length; i++)
            {
                if (p[i] != 0) state[i] = FGD;
                if (n[i] != 0 || b[i] != 0) state[i] = BGD;
            }
        }

        EnsureBothClasses(state, f.Width, f.Height, roiLocal, ctx);

        using var stateMat = MatInterop.FromBytes(state, f.Width, f.Height);
        using var work = new Mat();
        if (scale < 1) Cv2.Resize(stateMat, work, new Size(w, h), 0, 0, InterpolationFlags.Nearest); else stateMat.CopyTo(work);

        if (reuse && prev!.BackgroundModel.Length == 65)
        {
            bgdModel.Create(1, 65, MatType.CV_64FC1); fgdModel.Create(1, 65, MatType.CV_64FC1);
            Marshal.Copy(prev.BackgroundModel, 0, bgdModel.Data, 65);
            Marshal.Copy(prev.ForegroundModel, 0, fgdModel.Data, 65);
        }

        ctx.Token.ThrowIfCancellationRequested();
        Cv2.SetTheRNG(ctx.P.RandomSeed); // GMM k-means initialisation is otherwise non-deterministic
        int iterations = Math.Max(1, ctx.P.GrabCutIterations);
        // Iterate one step at a time so cancellation is honoured between iterations.
        for (int it = 0; it < iterations; it++)
        {
            ctx.Token.ThrowIfCancellationRequested();
            bool firstCall = it == 0 && !(reuse && prev!.BackgroundModel.Length == 65);
            Cv2.GrabCut(img, work, new Rect(), bgdModel, fgdModel, 1,
                firstCall ? GrabCutModes.InitWithMask : GrabCutModes.Eval);
        }

        // Back to full resolution.
        using var full = new Mat();
        if (scale < 1) Cv2.Resize(work, full, new Size(f.Width, f.Height), 0, 0, InterpolationFlags.Nearest); else work.CopyTo(full);
        var finalState = MatInterop.ToBytes(full);

        ctx.GrabCutState = new GrabCutState
        {
            Frame = f, Roi = ctx.Roi, StateMask = finalState, DownscaleFactor = scale,
            BackgroundModel = ToDoubles(bgdModel), ForegroundModel = ToDoubles(fgdModel),
        };
        ctx.Diagnostics["grabcut.frame"] = f.ToString();
        ctx.Diagnostics["grabcut.scale"] = scale.ToString("0.###", CultureInfo.InvariantCulture);
        ctx.Diagnostics["grabcut.reusedPreviousState"] = reuse.ToString();
        if (scale < 1) ctx.Warnings.Add($"ROI processed at {scale:P0} resolution (quality preset {ctx.P.Quality}); use 'Precise' for full resolution.");

        var fg = new byte[finalState.Length];
        for (int i = 0; i < fg.Length; i++) fg[i] = finalState[i] is FGD or PR_FGD ? (byte)1 : (byte)0;
        return MatInterop.FromBytes(fg, f.Width, f.Height);
    }

    /// <summary>GrabCut needs samples of both models; if the ROI fills the whole frame, use a thin ROI border as probable background.</summary>
    private static void EnsureBothClasses(byte[] state, int w, int h, Rect roi, SegmentationContext ctx)
    {
        bool hasBg = false, hasFg = false;
        foreach (var s in state) { if (s is BGD or PR_BGD) hasBg = true; else hasFg = true; }
        if (!hasFg) throw new SegmentationException("Background guidance covers the entire ROI; nothing left to segment.");
        if (hasBg) return;
        int band = Math.Max(1, Math.Min(roi.Width, roi.Height) / 20);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if ((x < band || y < band || x >= w - band || y >= h - band) && state[y * w + x] == PR_FGD)
                    state[y * w + x] = PR_BGD;
        ctx.Warnings.Add("ROI covers the full image extent; a border band was used as probable background.");
    }

    private static double[] ToDoubles(Mat m)
    {
        if (m.Empty()) return Array.Empty<double>();
        var d = new double[m.Total()];
        Marshal.Copy(m.Data, d, 0, d.Length);
        return d;
    }
}
