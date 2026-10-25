using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using OpenCvSharp;

namespace CuAnnotator.Vision.Segmentation;

/// <summary>
/// Morphological active contours (Márquez-Neila, Baumela &amp; Álvarez, IEEE TPAMI 36(1), 2014).
/// <para>
/// <b>RegionChanVese</b> (MorphACWE): with inside/outside mean colours c1, c0 over the Lab image I,
/// a pixel on the contour (|∇u| &gt; 0) joins the region when λ1‖I−c1‖² &lt; λ2‖I−c0‖².
/// <b>GeodesicEdge</b> (MorphGAC): edge-stopping g = 1/√(1+α|∇(Gσ*I)|); a balloon force dilates
/// (or erodes) where g exceeds its 40th percentile; the contour moves along ∇g·∇u.
/// Both modes apply the curvature operator (SI∘IS / IS∘SI) <c>smoothing</c> times per iteration and
/// stop when fewer than <c>ConvergenceTolerance</c>·|ROI| pixels change.
/// </para>
/// <para>Limitations: binary level-set, so topology changes are allowed but sub-pixel precision is not;
/// MorphACWE assumes two roughly homogeneous colour populations inside the ROI.</para>
/// </summary>
public sealed class ActiveContourSegmenter : SegmentationAlgorithmBase
{
    public override string Name => "Morphological Active Contour";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.ActiveContour;

    protected override RasterMask Run(SegmentationContext ctx)
    {
        var frame = ctx.Roi;
        using var pos = ctx.StrokeMask(ctx.Request.PositiveStrokes, frame);
        using var neg = ctx.StrokeMask(ctx.Request.NegativeStrokes, frame);
        using var init = InitialLevelSet(ctx, frame, pos);
        using var u = Evolve(ctx, frame, init, ctx.P.ActiveContourMode, ctx.P.ActiveContourIterations, pos, neg);
        ctx.AddStage("contour", MatInterop.MatToMask(u, frame), $"{ctx.P.ActiveContourMode} level set after evolution");
        foreach (var kv in MaskPostProcessor.Apply(u, PostProcessingOptions.From(ctx.P), null, pos, neg))
            ctx.Diagnostics[kv.Key] = kv.Value;
        return MatInterop.MatToMask(u, frame);
    }

    private static Mat InitialLevelSet(SegmentationContext ctx, PixelRect frame, Mat pos)
    {
        if (ctx.Request.InitialMask is { IsEmpty: false } im) { ctx.Diagnostics["ac.init"] = "initial mask"; return MatInterop.MaskToMat(im, frame); }
        var m = new Mat(frame.Height, frame.Width, MatType.CV_8UC1, Scalar.All(0));
        if (Cv2.CountNonZero(pos) > 0)
        {
            using var k = MaskPostProcessor.Disk(3);
            Cv2.Dilate(pos, m, k);
            ctx.Diagnostics["ac.init"] = "dilated foreground strokes";
        }
        else
        {
            int mx = Math.Max(1, frame.Width / 10), my = Math.Max(1, frame.Height / 10);
            m[new Rect(mx, my, Math.Max(1, frame.Width - 2 * mx), Math.Max(1, frame.Height - 2 * my))].SetTo(Scalar.All(1));
            ctx.Diagnostics["ac.init"] = "ROI shrunk by 10 %";
        }
        return m;
    }

    /// <summary>Evolves the 0/1 level set <paramref name="init"/> over <paramref name="frame"/>; returns a new Mat.</summary>
    internal static Mat Evolve(SegmentationContext ctx, PixelRect frame, Mat init, ActiveContourMode mode, int iterations, Mat? pos, Mat? neg)
    {
        int w = frame.Width, h = frame.Height, n = w * h;
        var u = MatInterop.ToBytes(init);
        for (int i = 0; i < n; i++) u[i] = u[i] != 0 ? (byte)1 : (byte)0;
        var posB = pos is null ? null : MatInterop.ToBytes(pos);
        var negB = neg is null ? null : MatInterop.ToBytes(neg);
        using var barrier = ctx.StrokeMask(ctx.Request.BarrierStrokes, frame);
        var barB = MatInterop.ToBytes(barrier);

        var lab = LabImage.FromBgr(ctx.Request.Image, frame);
        float[]? g = null, gx = null, gy = null; float gThreshold = 0;
        if (mode == ActiveContourMode.GeodesicEdge) (g, gx, gy, gThreshold) = EdgeStopping(ctx, frame);

        var gradMag = new bool[n];
        int it = 0, changed = 0;
        for (; it < Math.Max(1, iterations); it++)
        {
            ctx.Token.ThrowIfCancellationRequested();
            var before = (byte[])u.Clone();
            ContourIndicator(u, w, h, gradMag);

            if (mode == ActiveContourMode.RegionChanVese)
            {
                double[] c1 = new double[3], c0 = new double[3]; long n1 = 0, n0 = 0;
                for (int i = 0; i < n; i++)
                {
                    var acc = u[i] != 0 ? c1 : c0;
                    acc[0] += lab.Data[i * 3]; acc[1] += lab.Data[i * 3 + 1]; acc[2] += lab.Data[i * 3 + 2];
                    if (u[i] != 0) n1++; else n0++;
                }
                if (n1 == 0 || n0 == 0) break;
                for (int c = 0; c < 3; c++) { c1[c] /= n1; c0[c] /= n0; }
                double l1 = ctx.P.RegionLambdaInside, l2 = ctx.P.RegionLambdaOutside;
                for (int i = 0; i < n; i++)
                {
                    if (!gradMag[i]) continue;
                    double din = 0, dout = 0;
                    for (int c = 0; c < 3; c++)
                    {
                        double v = lab.Data[i * 3 + c];
                        din += (v - c1[c]) * (v - c1[c]);
                        dout += (v - c0[c]) * (v - c0[c]);
                    }
                    double aux = l1 * din - l2 * dout;
                    if (aux < 0) u[i] = 1; else if (aux > 0) u[i] = 0;
                }
            }
            else
            {
                if (ctx.P.Balloon != 0)
                {
                    using var um = MatInterop.FromBytes(u, w, h);
                    using var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
                    using var moved = new Mat();
                    if (ctx.P.Balloon > 0) Cv2.Dilate(um, moved, k); else Cv2.Erode(um, moved, k);
                    var mv = MatInterop.ToBytes(moved);
                    for (int i = 0; i < n; i++) if (g![i] > gThreshold) u[i] = mv[i];
                }
                // Image attachment: move along ∇g·∇u.
                var uCopy = (byte[])u.Clone();
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int i = y * w + x;
                        double dux = (x + 1 < w ? uCopy[i + 1] : uCopy[i]) - (x > 0 ? uCopy[i - 1] : uCopy[i]);
                        double duy = (y + 1 < h ? uCopy[i + w] : uCopy[i]) - (y > 0 ? uCopy[i - w] : uCopy[i]);
                        double aux = gx![i] * dux + gy![i] * duy;
                        if (aux > 0) u[i] = 1; else if (aux < 0) u[i] = 0;
                    }
            }

            for (int s = 0; s < ctx.P.ActiveContourSmoothing; s++) Curvature(u, w, h, (it + s) % 2 == 0);

            for (int i = 0; i < n; i++)
            {
                if (posB is not null && posB[i] != 0) u[i] = 1;
                if ((negB is not null && negB[i] != 0) || barB[i] != 0) u[i] = 0;
            }

            changed = 0;
            for (int i = 0; i < n; i++) if (u[i] != before[i]) changed++;
            if (changed <= ctx.P.ConvergenceTolerance * n) { it++; break; }
        }
        ctx.Diagnostics["ac.iterations"] = it.ToString();
        ctx.Diagnostics["ac.lastChangedPixels"] = changed.ToString();
        return MatInterop.FromBytes(u, w, h);
    }

    /// <summary>Marks pixels where the central-difference gradient of u is non-zero.</summary>
    private static void ContourIndicator(byte[] u, int w, int h, bool[] outFlags)
    {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                int dx = (x + 1 < w ? u[i + 1] : u[i]) - (x > 0 ? u[i - 1] : u[i]);
                int dy = (y + 1 < h ? u[i + w] : u[i]) - (y > 0 ? u[i - w] : u[i]);
                outFlags[i] = dx != 0 || dy != 0;
            }
    }

    private static readonly Mat[] LineElements = BuildLineElements();

    private static Mat[] BuildLineElements()
    {
        var els = new Mat[4];
        byte[][] patterns =
        {
            new byte[] { 0,0,0, 1,1,1, 0,0,0 },
            new byte[] { 0,1,0, 0,1,0, 0,1,0 },
            new byte[] { 1,0,0, 0,1,0, 0,0,1 },
            new byte[] { 0,0,1, 0,1,0, 1,0,0 },
        };
        for (int k = 0; k < 4; k++) els[k] = MatInterop.FromBytes(patterns[k], 3, 3);
        return els;
    }

    /// <summary>One application of SI∘IS (or IS∘SI when <paramref name="siFirst"/> is false).</summary>
    private static void Curvature(byte[] u, int w, int h, bool siFirst)
    {
        using var m = MatInterop.FromBytes(u, w, h);
        using var r = siFirst ? SI(IS(m)) : IS(SI(m));
        var d = MatInterop.ToBytes(r);
        Array.Copy(d, u, u.Length);
    }

    /// <summary>Sup-inf operator: max over the four line erosions.</summary>
    private static Mat SI(Mat m)
    {
        var acc = new Mat(m.Size(), MatType.CV_8UC1, Scalar.All(0));
        foreach (var k in LineElements)
        {
            using var e = new Mat();
            Cv2.Erode(m, e, k, null, 1, BorderTypes.Replicate);
            Cv2.Max(acc, e, acc);
        }
        return acc;
    }

    /// <summary>Inf-sup operator: min over the four line dilations.</summary>
    private static Mat IS(Mat m)
    {
        var acc = new Mat(m.Size(), MatType.CV_8UC1, Scalar.All(1));
        foreach (var k in LineElements)
        {
            using var d = new Mat();
            Cv2.Dilate(m, d, k, null, 1, BorderTypes.Replicate);
            Cv2.Min(acc, d, acc);
        }
        return acc;
    }

    private static (float[] g, float[] gx, float[] gy, float threshold) EdgeStopping(SegmentationContext ctx, PixelRect frame)
    {
        using var crop = new Mat(ctx.Request.Image, frame.ToRect());
        using var gray = new Mat();
        Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);
        using var f = new Mat();
        gray.ConvertTo(f, MatType.CV_32F, 1.0 / 255);
        if (ctx.P.EdgeSigma > 0) Cv2.GaussianBlur(f, f, new Size(0, 0), ctx.P.EdgeSigma);
        using var dx = new Mat(); using var dy = new Mat(); using var mag = new Mat();
        Cv2.Sobel(f, dx, MatType.CV_32F, 1, 0, 3, 0.125);
        Cv2.Sobel(f, dy, MatType.CV_32F, 0, 1, 3, 0.125);
        Cv2.Magnitude(dx, dy, mag);
        var magA = MatInterop.ToFloats(mag);
        var g = new float[magA.Length];
        for (int i = 0; i < g.Length; i++) g[i] = (float)(1.0 / Math.Sqrt(1.0 + ctx.P.EdgeAttraction * magA[i]));
        using var gm = new Mat(frame.Height, frame.Width, MatType.CV_32FC1);
        gm.SetArray(g);
        using var gdx = new Mat(); using var gdy = new Mat();
        Cv2.Sobel(gm, gdx, MatType.CV_32F, 1, 0, 3, 0.125);
        Cv2.Sobel(gm, gdy, MatType.CV_32F, 0, 1, 3, 0.125);
        var sorted = (float[])g.Clone();
        Array.Sort(sorted);
        float thr = sorted[(int)(0.4 * (sorted.Length - 1))];
        return (g, MatInterop.ToFloats(gdx), MatInterop.ToFloats(gdy), thr);
    }
}
