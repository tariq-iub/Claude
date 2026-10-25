using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.Morphology;

public sealed class PostProcessingOptions
{
    public bool MorphologicalCleanup { get; init; } = true;
    public int OpeningRadius { get; init; } = 1;
    public int ClosingRadius { get; init; } = 1;
    public bool PreserveHoles { get; init; } = true;
    public int MinComponentArea { get; init; } = 5;
    public bool KeepOnlySeededComponents { get; init; }
    public double BoundarySmoothing { get; init; }

    public static PostProcessingOptions From(Segmentation.SegmentationParameters p) => new()
    {
        MorphologicalCleanup = p.MorphologicalCleanup,
        OpeningRadius = p.OpeningRadius,
        ClosingRadius = p.ClosingRadius,
        PreserveHoles = p.PreserveHoles,
        MinComponentArea = p.MinComponentArea,
        KeepOnlySeededComponents = p.KeepOnlySeededComponents,
        BoundarySmoothing = p.BoundarySmoothing,
    };
}

/// <summary>Reusable binary-mask operations on 0/1 CV_8UC1 Mats (in-place where noted).</summary>
public static class MaskPostProcessor
{
    public static Mat Disk(int radius) =>
        Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(2 * radius + 1, 2 * radius + 1));

    public static void Open(Mat m, int radius)
    {
        if (radius <= 0) return;
        using var k = Disk(radius);
        Cv2.MorphologyEx(m, m, MorphTypes.Open, k);
    }

    public static void Close(Mat m, int radius)
    {
        if (radius <= 0) return;
        using var k = Disk(radius);
        // Pad so closing does not invent foreground at the frame border.
        using var padded = new Mat();
        Cv2.CopyMakeBorder(m, padded, radius, radius, radius, radius, BorderTypes.Constant, Scalar.All(0));
        Cv2.MorphologyEx(padded, padded, MorphTypes.Close, k);
        using var inner = new Mat(padded, new Rect(radius, radius, m.Cols, m.Rows));
        inner.CopyTo(m);
    }

    /// <summary>Fills all background regions not 4-connected to the frame border.</summary>
    public static void FillHoles(Mat m)
    {
        using var padded = new Mat();
        Cv2.CopyMakeBorder(m, padded, 1, 1, 1, 1, BorderTypes.Constant, Scalar.All(0));
        using var ff = padded.Clone();
        Cv2.FloodFill(ff, new Point(0, 0), Scalar.All(2), out _, null, null, FloodFillFlags.Link4);
        using var holes = new Mat();
        Cv2.Compare(ff, Scalar.All(0), holes, CmpTypes.EQ);
        padded.SetTo(Scalar.All(1), holes);
        using var inner = new Mat(padded, new Rect(1, 1, m.Cols, m.Rows));
        inner.CopyTo(m);
    }

    /// <summary>Removes 8-connected components with fewer than <paramref name="minArea"/> pixels. Returns the number removed.</summary>
    public static int RemoveSmallComponents(Mat m, int minArea)
    {
        if (minArea <= 1) return 0;
        using var labels = new Mat(); using var stats = new Mat(); using var cents = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
        var keep = new bool[n];
        int removed = 0;
        for (int i = 1; i < n; i++)
        {
            keep[i] = stats.At<int>(i, (int)ConnectedComponentsTypes.Area) >= minArea;
            if (!keep[i]) removed++;
        }
        if (removed > 0) ApplyKeep(m, labels, keep);
        return removed;
    }

    /// <summary>Keeps only components that intersect <paramref name="seeds"/>; if none do, keeps the largest.</summary>
    public static void KeepComponentsTouching(Mat m, Mat? seeds, bool fallbackToLargest = true)
    {
        using var labels = new Mat(); using var stats = new Mat(); using var cents = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
        if (n <= 2 && seeds is null) return;
        var keep = new bool[n];
        bool any = false;
        if (seeds is not null)
        {
            labels.GetArray(out int[] lab);
            var sd = MatInterop.ToBytes(seeds);
            for (int i = 0; i < lab.Length; i++)
                if (sd[i] != 0 && lab[i] > 0) { keep[lab[i]] = true; any = true; }
        }
        if (!any)
        {
            if (!fallbackToLargest) return;
            int best = -1, bestArea = 0;
            for (int i = 1; i < n; i++)
            {
                int a = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (a > bestArea) { bestArea = a; best = i; }
            }
            if (best > 0) keep[best] = true;
        }
        ApplyKeep(m, labels, keep);
    }

    /// <summary>Zeroes every pixel whose component label is not marked in <paramref name="keep"/>.</summary>
    private static void ApplyKeep(Mat m, Mat labels, bool[] keep)
    {
        labels.GetArray(out int[] lab);
        var data = MatInterop.ToBytes(m);
        for (int i = 0; i < lab.Length; i++)
            if (lab[i] > 0 && !keep[lab[i]]) data[i] = 0;
        System.Runtime.InteropServices.Marshal.Copy(data, 0, m.Data, data.Length);
    }

    public static void Smooth(Mat m, double sigma)
    {
        if (sigma <= 0) return;
        using var f = new Mat();
        m.ConvertTo(f, MatType.CV_32F);
        Cv2.GaussianBlur(f, f, new Size(0, 0), sigma);
        using var t = new Mat();
        Cv2.Threshold(f, t, 0.5, 1, ThresholdTypes.Binary);
        t.ConvertTo(m, MatType.CV_8U);
    }

    /// <summary>
    /// Standard post-processing chain. <paramref name="allowed"/> limits the result (e.g. the ROI);
    /// positive/negative seed masks are re-imposed at the end because they are hard constraints.
    /// </summary>
    public static Dictionary<string, string> Apply(Mat m, PostProcessingOptions o, Mat? allowed, Mat? positive, Mat? negative)
    {
        var diag = new Dictionary<string, string>();
        Cv2.Threshold(m, m, 0, 1, ThresholdTypes.Binary);
        if (o.MorphologicalCleanup) { Open(m, o.OpeningRadius); Close(m, o.ClosingRadius); }
        Smooth(m, o.BoundarySmoothing);
        if (!o.PreserveHoles) FillHoles(m);
        if (allowed is not null) Cv2.BitwiseAnd(m, allowed, m);
        int removed = RemoveSmallComponents(m, o.MinComponentArea);
        diag["post.removedSmallComponents"] = removed.ToString();
        if (o.KeepOnlySeededComponents && positive is not null) KeepComponentsTouching(m, positive, fallbackToLargest: false);
        if (positive is not null)
        {
            using var p = positive.Clone();
            if (allowed is not null) Cv2.BitwiseAnd(p, allowed, p);
            Cv2.BitwiseOr(m, p, m);
        }
        if (negative is not null) m.SetTo(Scalar.All(0), negative);
        return diag;
    }
}
