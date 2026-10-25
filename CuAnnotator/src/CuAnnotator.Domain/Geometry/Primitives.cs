namespace CuAnnotator.Domain.Geometry;

/// <summary>
/// Integer pixel index in canonical image space (origin top-left, X right, Y down, zero-based).
/// Pixel (X, Y) covers the continuous square [X, X+1) × [Y, Y+1); its centre is (X + 0.5, Y + 0.5).
/// </summary>
public readonly record struct PixelPoint(int X, int Y)
{
    public override string ToString() => $"({X}, {Y})";
}

/// <summary>Continuous position in image space (same axes as <see cref="PixelPoint"/>).</summary>
public readonly record struct ImagePoint(double X, double Y)
{
    /// <summary>Index of the pixel that contains this continuous point (floor rule).</summary>
    public PixelPoint ToPixel() => new((int)Math.Floor(X), (int)Math.Floor(Y));

    public static ImagePoint CenterOf(PixelPoint p) => new(p.X + 0.5, p.Y + 0.5);

    public override string ToString() => $"({X:0.###}, {Y:0.###})";
}

/// <summary>Axis-aligned rectangle of whole pixels. Right/Bottom are exclusive.</summary>
public readonly record struct PixelRect(int X, int Y, int Width, int Height)
{
    public int Right => X + Width;
    public int Bottom => Y + Height;
    public bool IsEmpty => Width <= 0 || Height <= 0;
    public long Area => IsEmpty ? 0 : (long)Width * Height;

    public static PixelRect Empty => new(0, 0, 0, 0);

    public static PixelRect FromCorners(int x0, int y0, int x1Exclusive, int y1Exclusive) =>
        new(Math.Min(x0, x1Exclusive), Math.Min(y0, y1Exclusive),
            Math.Abs(x1Exclusive - x0), Math.Abs(y1Exclusive - y0));

    /// <summary>Rectangle spanned by two pixels, both inclusive (e.g. a user drag).</summary>
    public static PixelRect FromInclusivePixels(PixelPoint a, PixelPoint b) =>
        new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Abs(a.X - b.X) + 1, Math.Abs(a.Y - b.Y) + 1);

    public bool Contains(int x, int y) => x >= X && y >= Y && x < Right && y < Bottom;
    public bool Contains(PixelPoint p) => Contains(p.X, p.Y);

    public PixelRect Intersect(PixelRect o)
    {
        int x0 = Math.Max(X, o.X), y0 = Math.Max(Y, o.Y);
        int x1 = Math.Min(Right, o.Right), y1 = Math.Min(Bottom, o.Bottom);
        return x1 <= x0 || y1 <= y0 ? Empty : new PixelRect(x0, y0, x1 - x0, y1 - y0);
    }

    public PixelRect Union(PixelRect o)
    {
        if (IsEmpty) return o;
        if (o.IsEmpty) return this;
        int x0 = Math.Min(X, o.X), y0 = Math.Min(Y, o.Y);
        return new PixelRect(x0, y0, Math.Max(Right, o.Right) - x0, Math.Max(Bottom, o.Bottom) - y0);
    }

    public PixelRect Inflate(int margin) => new(X - margin, Y - margin, Width + 2 * margin, Height + 2 * margin);

    /// <summary>Clip to the image extent [0,width) × [0,height).</summary>
    public PixelRect ClampTo(int width, int height) => Intersect(new PixelRect(0, 0, width, height));

    public override string ToString() => $"[{X}, {Y}, {Width}×{Height}]";
}
