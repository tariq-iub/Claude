namespace CuAnnotator.Domain.Geometry;

/// <summary>
/// One connected part of an annotation: an outer boundary and zero or more interior holes.
/// Vertices are the integer indices of boundary pixels (pixel-centre convention, see
/// documentation/04_Annotation_Data_Model.md). The rasterisation rule that turns a component back into
/// pixels is implemented in <c>CuAnnotator.Vision.Geometry.MaskRasterizer</c>.
/// </summary>
public sealed class PolygonComponent
{
    public List<PixelPoint> Outer { get; set; } = new();
    public List<List<PixelPoint>> Holes { get; set; } = new();

    public PolygonComponent() { }

    public PolygonComponent(IEnumerable<PixelPoint> outer, IEnumerable<IEnumerable<PixelPoint>>? holes = null)
    {
        Outer = outer.ToList();
        Holes = holes?.Select(h => h.ToList()).ToList() ?? new();
    }

    public int VertexCount => Outer.Count + Holes.Sum(h => h.Count);

    public PolygonComponent Clone() => new(Outer, Holes);

    public PolygonComponent Translate(int dx, int dy) =>
        new(Outer.Select(p => new PixelPoint(p.X + dx, p.Y + dy)),
            Holes.Select(h => h.Select(p => new PixelPoint(p.X + dx, p.Y + dy))));
}

public static class PolygonMath
{
    /// <summary>Shoelace signed area (positive for counter-clockwise in a Y-up frame; Y-down flips sign).</summary>
    public static double SignedArea(IReadOnlyList<PixelPoint> ring)
    {
        if (ring.Count < 3) return 0;
        double s = 0;
        for (int i = 0, j = ring.Count - 1; i < ring.Count; j = i++)
            s += (double)ring[j].X * ring[i].Y - (double)ring[i].X * ring[j].Y;
        return s / 2;
    }

    public static double Area(IReadOnlyList<PixelPoint> ring) => Math.Abs(SignedArea(ring));

    /// <summary>Geometric polygon area of a component (outer minus holes), in vertex units.</summary>
    public static double Area(PolygonComponent c) => Area(c.Outer) - c.Holes.Sum(h => Area(h));

    public static double Perimeter(IReadOnlyList<PixelPoint> ring, bool closed = true)
    {
        if (ring.Count < 2) return 0;
        double s = 0;
        int n = closed ? ring.Count : ring.Count - 1;
        for (int i = 0; i < n; i++)
        {
            var a = ring[i];
            var b = ring[(i + 1) % ring.Count];
            s += Math.Sqrt(Sq(b.X - a.X) + Sq(b.Y - a.Y));
        }
        return s;
    }

    public static PixelRect BoundingBox(IEnumerable<PixelPoint> pts)
    {
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = int.MinValue, y1 = int.MinValue;
        foreach (var p in pts)
        {
            x0 = Math.Min(x0, p.X); y0 = Math.Min(y0, p.Y);
            x1 = Math.Max(x1, p.X); y1 = Math.Max(y1, p.Y);
        }
        return x0 == int.MaxValue ? PixelRect.Empty : new PixelRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    /// <summary>True when two non-adjacent edges of the ring properly cross or overlap.</summary>
    public static bool IsSelfIntersecting(IReadOnlyList<PixelPoint> ring)
    {
        int n = ring.Count;
        if (n < 4) return false;
        for (int i = 0; i < n; i++)
        {
            var a1 = ring[i]; var a2 = ring[(i + 1) % n];
            for (int j = i + 1; j < n; j++)
            {
                if (j == i || (j + 1) % n == i || (i + 1) % n == j) continue; // adjacent edges share a vertex
                var b1 = ring[j]; var b2 = ring[(j + 1) % n];
                if (SegmentsIntersect(a1, a2, b1, b2)) return true;
            }
        }
        return false;
    }

    public static bool SegmentsIntersect(PixelPoint p1, PixelPoint p2, PixelPoint q1, PixelPoint q2)
    {
        long d1 = Cross(q1, q2, p1), d2 = Cross(q1, q2, p2), d3 = Cross(p1, p2, q1), d4 = Cross(p1, p2, q2);
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
        return (d1 == 0 && OnSegment(q1, q2, p1)) || (d2 == 0 && OnSegment(q1, q2, p2))
            || (d3 == 0 && OnSegment(p1, p2, q1)) || (d4 == 0 && OnSegment(p1, p2, q2));
    }

    private static long Cross(PixelPoint a, PixelPoint b, PixelPoint c) =>
        (long)(b.X - a.X) * (c.Y - a.Y) - (long)(b.Y - a.Y) * (c.X - a.X);

    private static bool OnSegment(PixelPoint a, PixelPoint b, PixelPoint p) =>
        Math.Min(a.X, b.X) <= p.X && p.X <= Math.Max(a.X, b.X) && Math.Min(a.Y, b.Y) <= p.Y && p.Y <= Math.Max(a.Y, b.Y);

    /// <summary>Ramer–Douglas–Peucker simplification of a closed ring with tolerance ε (pixels).</summary>
    public static List<PixelPoint> SimplifyClosed(IReadOnlyList<PixelPoint> ring, double epsilon)
    {
        if (ring.Count <= 4 || epsilon <= 0) return ring.ToList();
        // Split the ring at the vertex farthest from vertex 0 so both halves are open polylines.
        int far = 0; double best = -1;
        for (int i = 1; i < ring.Count; i++)
        {
            double d = Sq(ring[i].X - ring[0].X) + Sq(ring[i].Y - ring[0].Y);
            if (d > best) { best = d; far = i; }
        }
        var a = Rdp(ring.Take(far + 1).ToList(), epsilon);
        var b = Rdp(ring.Skip(far).Append(ring[0]).ToList(), epsilon);
        var result = new List<PixelPoint>(a.Count + b.Count);
        result.AddRange(a);
        result.AddRange(b.Skip(1).Take(b.Count - 2));
        return result.Count >= 3 ? result : ring.ToList();
    }

    private static List<PixelPoint> Rdp(List<PixelPoint> pts, double eps)
    {
        if (pts.Count < 3) return pts;
        var keep = new bool[pts.Count];
        keep[0] = keep[^1] = true;
        var stack = new Stack<(int, int)>();
        stack.Push((0, pts.Count - 1));
        while (stack.Count > 0)
        {
            var (s, e) = stack.Pop();
            double maxD = -1; int idx = -1;
            for (int i = s + 1; i < e; i++)
            {
                double d = DistanceToSegment(pts[i], pts[s], pts[e]);
                if (d > maxD) { maxD = d; idx = i; }
            }
            if (idx >= 0 && maxD > eps)
            {
                keep[idx] = true;
                stack.Push((s, idx));
                stack.Push((idx, e));
            }
        }
        return pts.Where((_, i) => keep[i]).ToList();
    }

    public static double DistanceToSegment(PixelPoint p, PixelPoint a, PixelPoint b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y;
        double len2 = dx * dx + dy * dy;
        if (len2 == 0) return Math.Sqrt(Sq(p.X - a.X) + Sq(p.Y - a.Y));
        double t = Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / len2, 0, 1);
        return Math.Sqrt(Sq(p.X - (a.X + t * dx)) + Sq(p.Y - (a.Y + t * dy)));
    }

    private static double Sq(double v) => v * v;
}
