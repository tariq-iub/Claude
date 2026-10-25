using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CuAnnotator.Desktop.Interaction;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Desktop.Rendering;

/// <summary>Pre-rendered display data for one annotation (rebuilt only when its mask or colour changes).</summary>
public sealed class AnnotationVisual
{
    public required Guid Id { get; init; }
    public required Color Color { get; init; }
    public required BitmapSource? Fill { get; init; }
    public required Rect FillRect { get; init; }
    public required Geometry Outline { get; init; }
    public required bool Visible { get; init; }
    public required int ClassId { get; init; }
}

/// <summary>Temporary tool graphics in image coordinates (ROI rubber band, brush strokes, lasso, cut line …).</summary>
public sealed class PreviewShape
{
    public required IReadOnlyList<ImagePoint> Points { get; init; }
    public bool Closed { get; init; }
    public Color Color { get; init; } = Colors.Yellow;
    /// <summary>Stroke width in image pixels (brushes) — if 0, <see cref="ScreenThickness"/> is used.</summary>
    public double ImageThickness { get; init; }
    public double ScreenThickness { get; init; } = 1.5;
    public bool Dashed { get; init; }
}

/// <summary>Everything the viewport draws. Owned by the editor view model; the viewport only reads it.</summary>
public sealed class ViewportScene
{
    public TiledImageSource? Image { get; set; }
    public List<AnnotationVisual> Annotations { get; } = new();
    public AnnotationVisual? Candidate { get; set; }
    public PixelRect? CandidateRoi { get; set; }
    public List<PreviewShape> Guidance { get; } = new();
    public List<PreviewShape> Preview { get; } = new();
    public HashSet<Guid> Selected { get; } = new();
    /// <summary>Polygon being vertex-edited (image coordinates of boundary-pixel indices).</summary>
    public List<PolygonComponent>? EditPolygons { get; set; }
    public (int Component, int Ring, int Index)? HotVertex { get; set; }
    public OverlayMode Mode { get; set; } = OverlayMode.Normal;
    public double Opacity { get; set; } = 0.45;
    public bool ShowBoundaries { get; set; } = true;
    public bool ShowMasks { get; set; } = true;
    public bool ShowGrid { get; set; } = true;

    public event EventHandler? Changed;
    public void Invalidate() => Changed?.Invoke(this, EventArgs.Empty);
}

/// <summary>Creates mask bitmaps and contour geometry for display.</summary>
public static class OverlayFactory
{
    private const long MaxFullResPixels = 4_000_000;

    public static Color ParseColor(string hex)
    {
        try { return (Color)ColorConverter.ConvertFromString(hex); }
        catch (FormatException) { return Colors.Magenta; }
    }

    /// <summary>
    /// Premultiplied BGRA bitmap of the mask over its bounds. Masks larger than 4 MP are rendered at reduced
    /// resolution (nearest neighbour) for display only; exports always use the full-resolution mask.
    /// </summary>
    public static (BitmapSource? Bitmap, Rect Rect) MaskBitmap(RasterMask mask, Color color)
    {
        var b = mask.Bounds;
        if (b.IsEmpty) return (null, Rect.Empty);
        int step = 1;
        while ((long)(b.Width / step) * (b.Height / step) > MaxFullResPixels) step *= 2;
        int w = Math.Max(1, b.Width / step), h = Math.Max(1, b.Height / step);
        var px = new byte[w * h * 4];
        for (int y = 0; y < h; y++)
        {
            int row = Math.Min(b.Height - 1, y * step) * b.Width;
            for (int x = 0; x < w; x++)
            {
                if (mask.Data[row + Math.Min(b.Width - 1, x * step)] == 0) continue;
                int i = (y * w + x) * 4;
                px[i] = color.B; px[i + 1] = color.G; px[i + 2] = color.R; px[i + 3] = 255;
            }
        }
        var bmp = BitmapSource.Create(w, h, 96, 96, PixelFormats.Pbgra32, null, px, w * 4);
        bmp.Freeze();
        return (bmp, new Rect(b.X, b.Y, b.Width, b.Height));
    }

    /// <summary>Outline through boundary pixel centres (x + 0.5, y + 0.5) of every ring.</summary>
    public static Geometry Outline(IEnumerable<PolygonComponent> comps)
    {
        var g = new StreamGeometry { FillRule = FillRule.EvenOdd };
        using (var ctx = g.Open())
            foreach (var c in comps)
                foreach (var ring in c.Holes.Prepend(c.Outer))
                {
                    if (ring.Count == 0) continue;
                    ctx.BeginFigure(new Point(ring[0].X + 0.5, ring[0].Y + 0.5), false, true);
                    if (ring.Count == 1) ctx.LineTo(new Point(ring[0].X + 0.5, ring[0].Y + 0.5), true, false);
                    else ctx.PolyLineTo(ring.Skip(1).Select(p => new Point(p.X + 0.5, p.Y + 0.5)).ToList(), true, false);
                }
        g.Freeze();
        return g;
    }

    public static AnnotationVisual Build(Annotation a, Taxonomy taxonomy)
    {
        var color = ParseColor(taxonomy.Find(a.ClassId)?.ColorHex ?? "#FF00FF");
        var (bmp, rect) = MaskBitmap(a.Mask, color);
        return new AnnotationVisual { Id = a.Id, Color = color, Fill = bmp, FillRect = rect, Outline = Outline(a.Polygons), Visible = a.IsVisible, ClassId = a.ClassId };
    }
}
