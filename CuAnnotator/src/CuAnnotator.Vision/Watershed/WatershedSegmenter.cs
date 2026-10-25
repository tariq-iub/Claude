using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Vision.Watershed;

/// <summary>
/// Marker-controlled watershed (Meyer's flooding, OpenCV implementation) with exactly two marker
/// labels — background (1) and foreground (2) — which prevents uncontrolled over-segmentation.
/// <para>
/// Background markers: everything in the context frame outside the ROI plus background strokes
/// (and a thin ROI border if the ROI touches every frame edge). Foreground markers: foreground strokes;
/// else the eroded initial mask; else automatic markers = pixels in the inner half of the ROI within
/// ΔE00 ≤ <c>WatershedAutoMarkerDeltaE</c> of the central reference colour, eroded by 2 px.
/// The flooding relief is the colour gradient of the Gaussian-smoothed image. Watershed lines (−1)
/// are assigned to background.
/// </para>
/// </summary>
public sealed class WatershedSegmenter : SegmentationAlgorithmBase
{
    public override string Name => "Marker-Controlled Watershed";
    public override string Version => "1.0.0";
    public override SegmentationAlgorithmKind Kind => SegmentationAlgorithmKind.Watershed;

    protected override RasterMask Run(SegmentationContext ctx)
    {
        var frame = ctx.ContextFrame(Math.Max(ctx.P.RoiContextMargin, 0.05));
        using var crop = new Mat(ctx.Request.Image, frame.ToRect());
        using var img = crop.Clone();
        if (ctx.P.WatershedPreBlurSigma > 0) Cv2.GaussianBlur(img, img, new Size(0, 0), ctx.P.WatershedPreBlurSigma);

        using var allowed = GrabCut.GrabCutSegmenter.RoiMask(ctx, frame);
        using var pos = ctx.StrokeMask(ctx.Request.PositiveStrokes, frame);
        using var neg = ctx.StrokeMask(ctx.Request.NegativeStrokes, frame);
        using var bar = ctx.StrokeMask(ctx.Request.BarrierStrokes, frame);

        var markers = new int[frame.Width * frame.Height];
        var allowedB = MatInterop.ToBytes(allowed);
        bool anyOutside = false;
        for (int i = 0; i < markers.Length; i++) if (allowedB[i] == 0) { markers[i] = 1; anyOutside = true; }
        if (!anyOutside)
        {
            int w = frame.Width, h = frame.Height;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) markers[y * w + x] = 1;
        }

        using var fgMarker = ForegroundMarkers(ctx, frame, pos);
        var fgB = MatInterop.ToBytes(fgMarker);
        var negB = MatInterop.ToBytes(neg); var barB = MatInterop.ToBytes(bar);
        int fgCount = 0;
        for (int i = 0; i < markers.Length; i++)
        {
            if (fgB[i] != 0 && allowedB[i] != 0) { markers[i] = 2; fgCount++; }
            if (negB[i] != 0) markers[i] = 1;
        }
        if (fgCount == 0) throw new SegmentationException("Watershed needs at least one foreground marker inside the ROI.");

        using var markerMat = new Mat(frame.Height, frame.Width, MatType.CV_32SC1);
        markerMat.SetArray(markers);
        // Barrier strokes are drawn as unlabeled ridges with maximal relief so flooding cannot cross easily.
        if (barB.Any(b => b != 0)) img.SetTo(Scalar.All(255), bar);
        ctx.Token.ThrowIfCancellationRequested();
        Cv2.Watershed(img, markerMat);
        markerMat.GetArray(out int[] labels);

        var fg = new byte[labels.Length];
        for (int i = 0; i < fg.Length; i++) fg[i] = labels[i] == 2 ? (byte)1 : (byte)0;
        using var m = MatInterop.FromBytes(fg, frame.Width, frame.Height);
        ctx.AddStage("watershed", MatInterop.MatToMask(m, frame), "Foreground basin after flooding");
        ctx.Diagnostics["watershed.foregroundMarkerPixels"] = fgCount.ToString();
        foreach (var kv in MaskPostProcessor.Apply(m, PostProcessingOptions.From(ctx.P), allowed, pos, neg))
            ctx.Diagnostics[kv.Key] = kv.Value;
        return MatInterop.MatToMask(m, frame);
    }

    private static Mat ForegroundMarkers(SegmentationContext ctx, PixelRect frame, Mat pos)
    {
        if (Cv2.CountNonZero(pos) > 0) { ctx.Diagnostics["watershed.markers"] = "foreground strokes"; return pos.Clone(); }
        if (ctx.Request.InitialMask is { IsEmpty: false } init)
        {
            var m = MatInterop.MaskToMat(init, frame);
            using var k = MaskPostProcessor.Disk(2);
            using var eroded = new Mat();
            Cv2.Erode(m, eroded, k);
            if (Cv2.CountNonZero(eroded) > 0) eroded.CopyTo(m);
            ctx.Diagnostics["watershed.markers"] = "eroded initial mask";
            return m;
        }
        // Automatic markers from colour similarity in the inner half of the ROI.
        var roi = ctx.Roi;
        var inner = new PixelRect(roi.X + roi.Width / 4, roi.Y + roi.Height / 4, Math.Max(1, roi.Width / 2), Math.Max(1, roi.Height / 2));
        var lab = LabImage.FromBgr(ctx.Request.Image, inner);
        var centre = new byte[inner.Width * inner.Height];
        int cw = Math.Max(1, inner.Width / 2), ch = Math.Max(1, inner.Height / 2);
        for (int y = (inner.Height - ch) / 2; y < (inner.Height + ch) / 2; y++)
            for (int x = (inner.Width - cw) / 2; x < (inner.Width + cw) / 2; x++) centre[y * inner.Width + x] = 1;
        var reference = lab.Median(centre)!.Value;
        var d = lab.DistanceMap(reference, ColorDifferenceMetric.CIEDE2000);
        var sel = new byte[d.Length];
        for (int i = 0; i < d.Length; i++) sel[i] = d[i] <= ctx.P.WatershedAutoMarkerDeltaE ? (byte)1 : (byte)0;
        var auto = MatInterop.MaskToMat(new RasterMask(inner, sel), frame);
        using (var k = MaskPostProcessor.Disk(2))
        using (var eroded = new Mat())
        {
            Cv2.Erode(auto, eroded, k);
            if (Cv2.CountNonZero(eroded) > 0) eroded.CopyTo(auto);
        }
        MaskPostProcessor.KeepComponentsTouching(auto, null, fallbackToLargest: true);
        ctx.Diagnostics["watershed.markers"] = $"automatic (ΔE00 ≤ {ctx.P.WatershedAutoMarkerDeltaE} of {reference})";
        return auto;
    }
}
