using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Contours;
using CuAnnotator.Vision.Geometry;

namespace CuAnnotator.Application.Annotations;

using Annotation = Domain.Entities.Annotation;

/// <summary>Keeps the vector view and derived measurements of an annotation consistent with its mask.</summary>
public static class AnnotationGeometryService
{
    /// <summary>Sets a new authoritative mask and regenerates polygons (exact, no simplification) and measurements.</summary>
    public static void SetMask(Annotation a, RasterMask mask)
    {
        a.Mask = mask.Trim();
        a.Polygons = ContourExtractor.Extract(a.Mask);
        RefreshMeasurements(a);
    }

    /// <summary>
    /// Applies an edited polygon geometry: the mask is regenerated with the documented
    /// rasterisation rule (<see cref="MaskRasterizer"/>) and becomes the new authority.
    /// The edited polygons are kept as the vector view.
    /// </summary>
    public static void SetPolygons(Annotation a, IReadOnlyList<PolygonComponent> polygons, int imageWidth, int imageHeight)
    {
        a.Polygons = polygons.Select(p => p.Clone()).ToList();
        a.Mask = MaskRasterizer.Rasterize(a.Polygons, imageWidth, imageHeight);
        RefreshMeasurements(a);
    }

    public static void RefreshMeasurements(Annotation a)
    {
        a.PixelArea = a.Mask.PixelCount;
        a.BoundingBox = a.Mask.ContentBounds();
        var (cx, cy) = a.Mask.Centroid();
        a.CentroidX = cx; a.CentroidY = cy;
        a.Perimeter = a.Polygons.Sum(c => PolygonMath.Perimeter(c.Outer) + c.Holes.Sum(h => PolygonMath.Perimeter(h)));
    }
}
