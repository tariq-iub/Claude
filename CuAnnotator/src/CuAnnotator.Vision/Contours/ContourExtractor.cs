using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.Contours;

/// <summary>
/// Mask → editable polygon components using OpenCV border following (Suzuki–Abe) with
/// <c>RETR_CCOMP</c>: top-level contours are the outer boundaries of 8-connected foreground
/// components, second-level contours are their holes. With <c>CHAIN_APPROX_NONE</c> every
/// boundary pixel is a vertex, so <see cref="Geometry.MaskRasterizer"/> reproduces the mask exactly.
/// </summary>
public static class ContourExtractor
{
    public static List<PolygonComponent> Extract(RasterMask mask, double simplifyEpsilon = 0)
    {
        var result = new List<PolygonComponent>();
        if (mask.Bounds.IsEmpty || mask.IsEmpty) return result;
        // Pad by one pixel so contours touching the frame edge are traced correctly.
        var frame = mask.Bounds.Inflate(1);
        using var m = MatInterop.MaskToMat(mask, frame);
        Cv2.FindContours(m, out Point[][] contours, out HierarchyIndex[] hierarchy,
            RetrievalModes.CComp, ContourApproximationModes.ApproxNone, new Point(frame.X, frame.Y));

        for (int i = 0; i < contours.Length; i++)
        {
            if (hierarchy[i].Parent != -1) continue; // holes are collected through their parent
            var comp = new PolygonComponent { Outer = ToRing(contours[i], simplifyEpsilon) };
            for (int c = hierarchy[i].Child; c != -1; c = hierarchy[c].Next)
                comp.Holes.Add(ToRing(contours[c], simplifyEpsilon));
            result.Add(comp);
        }
        // Deterministic order: largest outer ring first, then by position.
        return result.OrderByDescending(c => PolygonMath.Area(c.Outer))
                     .ThenBy(c => c.Outer.Min(p => p.Y)).ThenBy(c => c.Outer.Min(p => p.X)).ToList();
    }

    private static List<PixelPoint> ToRing(Point[] pts, double eps)
    {
        var ring = pts.Select(p => new PixelPoint(p.X, p.Y)).ToList();
        return eps > 0 ? PolygonMath.SimplifyClosed(ring, eps) : ring;
    }
}
