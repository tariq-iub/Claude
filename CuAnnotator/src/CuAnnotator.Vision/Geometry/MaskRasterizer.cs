using CuAnnotator.Domain.Geometry;
using OpenCvSharp;

namespace CuAnnotator.Vision.Geometry;

/// <summary>
/// Polygon components → raster mask. <b>Rasterisation rule</b> (documented in 04_Annotation_Data_Model.md):
/// <list type="number">
/// <item>Vertices are integer pixel indices (pixel centres).</item>
/// <item>Components are processed largest-outer-first so islands inside holes are drawn after the hole.</item>
/// <item>The outer ring is filled (scan-line fill, 8-connected edges) and its boundary pixels are set.</item>
/// <item>Each hole ring is filled with 0 and then its boundary pixels are set to 1 again — hole rings
/// produced by border following run through foreground pixels, so the boundary belongs to the region.</item>
/// </list>
/// For rings produced by <see cref="Contours.ContourExtractor"/> without simplification this
/// reproduces the source mask exactly (verified by round-trip tests).
/// </summary>
public static class MaskRasterizer
{
    public static RasterMask Rasterize(IEnumerable<PolygonComponent> components, int imageWidth, int imageHeight)
    {
        var comps = components.Where(c => c.Outer.Count > 0).ToList();
        if (comps.Count == 0) return RasterMask.Empty;
        var all = comps.SelectMany(c => c.Outer.Concat(c.Holes.SelectMany(h => h)));
        var frame = PolygonMath.BoundingBox(all).Intersect(new PixelRect(0, 0, imageWidth, imageHeight));
        if (frame.IsEmpty) return RasterMask.Empty;
        var bb = PolygonMath.BoundingBox(all);
        using var m = new Mat(bb.Height, bb.Width, MatType.CV_8UC1, Scalar.All(0));
        foreach (var c in comps.OrderByDescending(c => PolygonMath.Area(c.Outer)))
        {
            var outer = Local(c.Outer, bb);
            Cv2.FillPoly(m, new[] { outer }, Scalar.All(1), LineTypes.Link8);
            DrawRing(m, outer, 1);
            foreach (var h in c.Holes)
            {
                var hole = Local(h, bb);
                Cv2.FillPoly(m, new[] { hole }, Scalar.All(0), LineTypes.Link8);
                DrawRing(m, hole, 1);
            }
        }
        var full = new RasterMask(bb, Interop.MatInterop.ToBytes(m));
        return full.Reframe(frame).Trim();
    }

    private static Point[] Local(List<PixelPoint> ring, PixelRect bb) =>
        ring.Select(p => new Point(p.X - bb.X, p.Y - bb.Y)).ToArray();

    private static void DrawRing(Mat m, Point[] ring, byte v)
    {
        if (ring.Length == 1) { m.Set(ring[0].Y, ring[0].X, v); return; }
        Cv2.Polylines(m, new[] { ring }, true, Scalar.All(v), 1, LineTypes.Link8);
    }
}
