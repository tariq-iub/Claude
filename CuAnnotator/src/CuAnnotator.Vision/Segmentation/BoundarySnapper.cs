using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.Segmentation;

/// <summary>
/// Smart boundary brush: moves each point of an approximate user path along its local normal
/// (within ±<c>searchRadius</c> px) to the position of maximal smoothed gradient magnitude, with a
/// small distance penalty so the path prefers nearby edges, then median-filters the offsets to
/// suppress jumps between unrelated edges. The snapped path is returned for user editing and can
/// be used as a barrier constraint or as a cut line.
/// </summary>
public static class BoundarySnapper
{
    public static List<PixelPoint> Snap(Mat bgrImage, IReadOnlyList<PixelPoint> path, int searchRadius = 6, double distancePenalty = 0.02)
    {
        if (path.Count < 2 || searchRadius <= 0) return path.ToList();
        var pts = Resample(path, 2.0);
        var bb = PolygonMath.BoundingBox(pts).Inflate(searchRadius + 3).ClampTo(bgrImage.Width, bgrImage.Height);
        if (bb.IsEmpty) return path.ToList();

        using var crop = new Mat(bgrImage, bb.ToRect());
        using var gray = new Mat();
        Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);
        using var f = new Mat();
        gray.ConvertTo(f, MatType.CV_32F, 1.0 / 255);
        Cv2.GaussianBlur(f, f, new Size(0, 0), 1.0);
        using var dx = new Mat(); using var dy = new Mat(); using var mag = new Mat();
        Cv2.Sobel(f, dx, MatType.CV_32F, 1, 0, 3);
        Cv2.Sobel(f, dy, MatType.CV_32F, 0, 1, 3);
        Cv2.Magnitude(dx, dy, mag);
        var g = MatInterop.ToFloats(mag);
        float max = g.Length == 0 ? 1 : Math.Max(1e-6f, g.Max());

        var offsets = new double[pts.Count];
        var normals = new (double X, double Y)[pts.Count];
        for (int i = 0; i < pts.Count; i++)
        {
            var a = pts[Math.Max(0, i - 1)]; var b = pts[Math.Min(pts.Count - 1, i + 1)];
            double tx = b.X - a.X, ty = b.Y - a.Y, len = Math.Sqrt(tx * tx + ty * ty);
            normals[i] = len == 0 ? (0, 0) : (-ty / len, tx / len);
            double best = double.NegativeInfinity, bestT = 0;
            for (int t = -searchRadius; t <= searchRadius; t++)
            {
                int x = (int)Math.Round(pts[i].X + normals[i].X * t) - bb.X;
                int y = (int)Math.Round(pts[i].Y + normals[i].Y * t) - bb.Y;
                if (x < 0 || y < 0 || x >= bb.Width || y >= bb.Height) continue;
                double score = g[y * bb.Width + x] / max - distancePenalty * Math.Abs(t);
                if (score > best) { best = score; bestT = t; }
            }
            offsets[i] = bestT;
        }
        // Median filter (window 5) for robustness.
        var smoothed = new double[offsets.Length];
        for (int i = 0; i < offsets.Length; i++)
        {
            var win = new List<double>();
            for (int k = -2; k <= 2; k++) if (i + k >= 0 && i + k < offsets.Length) win.Add(offsets[i + k]);
            win.Sort();
            smoothed[i] = win[win.Count / 2];
        }
        var result = new List<PixelPoint>(pts.Count);
        for (int i = 0; i < pts.Count; i++)
        {
            var p = new PixelPoint((int)Math.Round(pts[i].X + normals[i].X * smoothed[i]), (int)Math.Round(pts[i].Y + normals[i].Y * smoothed[i]));
            if (result.Count == 0 || result[^1] != p) result.Add(p);
        }
        return result;
    }

    /// <summary>Resamples a polyline at approximately uniform spacing.</summary>
    public static List<PixelPoint> Resample(IReadOnlyList<PixelPoint> path, double spacing)
    {
        var res = new List<PixelPoint> { path[0] };
        double carry = 0;
        for (int i = 1; i < path.Count; i++)
        {
            double x0 = path[i - 1].X, y0 = path[i - 1].Y, x1 = path[i].X, y1 = path[i].Y;
            double len = Math.Sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
            double t = spacing - carry;
            while (t <= len)
            {
                var p = new PixelPoint((int)Math.Round(x0 + (x1 - x0) * t / len), (int)Math.Round(y0 + (y1 - y0) * t / len));
                if (res[^1] != p) res.Add(p);
                t += spacing;
            }
            carry = len - (t - spacing);
        }
        if (res[^1] != path[^1]) res.Add(path[^1]);
        return res;
    }
}
