using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.GrabCut;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Morphology;
using CuAnnotator.Vision.Watershed;
using OpenCvSharp;

namespace CuAnnotator.Vision.Segmentation;

/// <summary>Registry of segmentation algorithms plus convenience operations (magic wand, colour select).</summary>
public sealed class SegmentationEngine
{
    private readonly Dictionary<SegmentationAlgorithmKind, ISegmentationAlgorithm> _algorithms;

    public SegmentationEngine(IEnumerable<ISegmentationAlgorithm>? algorithms = null)
    {
        _algorithms = (algorithms ?? DefaultAlgorithms()).ToDictionary(a => a.Kind);
    }

    public static IEnumerable<ISegmentationAlgorithm> DefaultAlgorithms() => new ISegmentationAlgorithm[]
    {
        new GrabCutSegmenter(), new LabColorSegmenter(), new WatershedSegmenter(),
        new ActiveContourSegmenter(), new HybridSegmenter(),
    };

    public IReadOnlyCollection<ISegmentationAlgorithm> Algorithms => _algorithms.Values;

    public ISegmentationAlgorithm Get(SegmentationAlgorithmKind kind) =>
        _algorithms.TryGetValue(kind, out var a) ? a : throw new NotSupportedException($"No algorithm registered for {kind}.");

    public Task<SegmentationResult> SegmentAsync(SegmentationRequest request, CancellationToken ct) =>
        Get(request.Parameters.Algorithm).SegmentAsync(request, ct);

    /// <summary>
    /// Magic wand: connected CIELAB region growing from a single click, limited to a window of
    /// ±<paramref name="maxRadius"/> pixels around the seed.
    /// </summary>
    public Task<SegmentationResult> MagicWandAsync(Mat image, PixelPoint seed, SegmentationParameters p, int maxRadius, CancellationToken ct)
    {
        var roi = new PixelRect(seed.X - maxRadius, seed.Y - maxRadius, 2 * maxRadius + 1, 2 * maxRadius + 1).ClampTo(image.Width, image.Height);
        var pp = p.Clone();
        pp.Algorithm = SegmentationAlgorithmKind.LabColor;
        pp.RequireConnectivity = true;
        return Get(SegmentationAlgorithmKind.LabColor).SegmentAsync(new SegmentationRequest
        {
            Image = image, Roi = roi, Parameters = pp, PositiveStrokes = new[] { Stroke.Point(seed, 1) },
        }, ct);
    }

    /// <summary>Colour-based selection: all pixels in the ROI similar to the sampled colour (not necessarily connected).</summary>
    public Task<SegmentationResult> ColorSelectAsync(Mat image, PixelRect roi, IReadOnlyList<Stroke> samples, SegmentationParameters p, CancellationToken ct)
    {
        var pp = p.Clone();
        pp.Algorithm = SegmentationAlgorithmKind.LabColor;
        pp.RequireConnectivity = false;
        return Get(SegmentationAlgorithmKind.LabColor).SegmentAsync(new SegmentationRequest
        {
            Image = image, Roi = roi, Parameters = pp, PositiveStrokes = samples,
        }, ct);
    }

    /// <summary>
    /// Splits a mask along a cut polyline (drawn 1 px wide, 8-connected) into its connected parts.
    /// Cut pixels are assigned to the adjacent part with the most neighbours so no pixel is lost.
    /// </summary>
    public static List<RasterMask> Split(RasterMask mask, IReadOnlyList<PixelPoint> cut)
    {
        if (mask.IsEmpty) return new();
        var frame = mask.Bounds;
        using var m = MatInterop.MaskToMat(mask, frame);
        using var cutMat = new Mat(frame.Height, frame.Width, MatType.CV_8UC1, Scalar.All(0));
        MatInterop.PaintStrokes(cutMat, new[] { new Stroke(cut, 0) }, frame, 1);
        using var cutOnMask = new Mat();
        Cv2.BitwiseAnd(cutMat, m, cutOnMask);
        using var parts = m.Clone();
        parts.SetTo(Scalar.All(0), cutMat);
        using var labels = new Mat();
        int n = Cv2.ConnectedComponents(parts, labels, PixelConnectivity.Connectivity4, MatType.CV_32S);
        labels.GetArray(out int[] lab);
        var cutB = MatInterop.ToBytes(cutOnMask);
        int w = frame.Width, h = frame.Height;
        // Assign cut pixels to a neighbouring label (iterate until stable).
        bool changedAny = true;
        while (changedAny)
        {
            changedAny = false;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = y * w + x;
                    if (cutB[i] == 0 || lab[i] != 0) continue;
                    var votes = new Dictionary<int, int>();
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                            int l = lab[yy * w + xx];
                            if (l > 0) votes[l] = votes.GetValueOrDefault(l) + 1;
                        }
                    if (votes.Count == 0) continue;
                    lab[i] = votes.OrderByDescending(v => v.Value).ThenBy(v => v.Key).First().Key;
                    changedAny = true;
                }
        }
        var result = new List<RasterMask>();
        for (int l = 1; l < n; l++)
        {
            var d = new byte[lab.Length];
            for (int i = 0; i < d.Length; i++) d[i] = lab[i] == l ? (byte)1 : (byte)0;
            var rm = new RasterMask(frame, d).Trim();
            if (!rm.IsEmpty) result.Add(rm);
        }
        return result;
    }

    /// <summary>Fills interior holes of a mask (Fill Region tool).</summary>
    public static RasterMask FillHoles(RasterMask mask)
    {
        if (mask.IsEmpty) return mask;
        using var m = MatInterop.MaskToMat(mask, mask.Bounds);
        MaskPostProcessor.FillHoles(m);
        return MatInterop.MatToMask(m, mask.Bounds);
    }

    /// <summary>Paints (add) or erases a brush stroke on a mask.</summary>
    public static RasterMask Paint(RasterMask mask, Stroke stroke, bool add, int imageWidth, int imageHeight)
    {
        var sb = PolygonMath.BoundingBox(stroke.Points).Inflate(stroke.Radius + 1);
        var frame = mask.Bounds.Union(sb).ClampTo(imageWidth, imageHeight);
        if (frame.IsEmpty) return mask;
        using var m = MatInterop.MaskToMat(mask, frame);
        MatInterop.PaintStrokes(m, new[] { stroke }, frame, add ? (byte)1 : (byte)0);
        return MatInterop.MatToMask(m, frame);
    }
}
