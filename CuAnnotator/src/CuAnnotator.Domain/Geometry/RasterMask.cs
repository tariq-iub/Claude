namespace CuAnnotator.Domain.Geometry;

/// <summary>
/// Authoritative binary geometry of an annotation: a 0/1 byte raster stored only over its
/// <see cref="Bounds"/> in image coordinates (row-major). Pixels outside the bounds are 0.
/// Instances are treated as immutable by convention: every edit produces a new mask so that
/// undo/redo can hold references safely.
/// </summary>
public sealed class RasterMask : IEquatable<RasterMask>
{
    public PixelRect Bounds { get; }
    public byte[] Data { get; }

    public int Width => Bounds.Width;
    public int Height => Bounds.Height;

    public RasterMask(PixelRect bounds, byte[]? data = null)
    {
        if (bounds.Width < 0 || bounds.Height < 0) throw new ArgumentOutOfRangeException(nameof(bounds));
        Bounds = bounds;
        Data = data ?? new byte[bounds.Width * bounds.Height];
        if (Data.Length != bounds.Width * bounds.Height)
            throw new ArgumentException($"Mask data length {Data.Length} does not match bounds {bounds}.");
    }

    public static RasterMask Empty { get; } = new(PixelRect.Empty);

    public bool Get(int x, int y) => Bounds.Contains(x, y) && Data[(y - Bounds.Y) * Width + (x - Bounds.X)] != 0;

    public long PixelCount
    {
        get
        {
            long n = 0;
            foreach (var b in Data) if (b != 0) n++;
            return n;
        }
    }

    public bool IsEmpty => PixelCount == 0;

    /// <summary>Tight bounding box of the set pixels.</summary>
    public PixelRect ContentBounds()
    {
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
        for (int y = 0; y < Height; y++)
        {
            int row = y * Width;
            for (int x = 0; x < Width; x++)
            {
                if (Data[row + x] == 0) continue;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                y1 = y;
            }
        }
        return x1 < 0 ? PixelRect.Empty : new PixelRect(Bounds.X + x0, Bounds.Y + y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    /// <summary>Copy of this mask re-expressed over <paramref name="target"/> (cropping or padding).</summary>
    public RasterMask Reframe(PixelRect target)
    {
        var result = new RasterMask(target);
        var overlap = Bounds.Intersect(target);
        for (int y = overlap.Y; y < overlap.Bottom; y++)
            Array.Copy(Data, (y - Bounds.Y) * Width + (overlap.X - Bounds.X),
                       result.Data, (y - target.Y) * target.Width + (overlap.X - target.X), overlap.Width);
        return result;
    }

    /// <summary>Shrinks the bounds to the set pixels (an empty mask becomes <see cref="Empty"/>).</summary>
    public RasterMask Trim()
    {
        var cb = ContentBounds();
        return cb.IsEmpty ? Empty : cb == Bounds ? this : Reframe(cb);
    }

    public RasterMask ClipToImage(int width, int height) => Reframe(Bounds.ClampTo(width, height)).Trim();

    public RasterMask Union(RasterMask other) => Combine(other, Bounds.Union(other.Bounds), (a, b) => a | b);
    public RasterMask Intersect(RasterMask other) => Combine(other, Bounds.Intersect(other.Bounds), (a, b) => a & b);
    public RasterMask Subtract(RasterMask other) => Combine(other, Bounds, (a, b) => a & ~b & 1);

    private RasterMask Combine(RasterMask other, PixelRect frame, Func<int, int, int> op)
    {
        if (frame.IsEmpty) return Empty;
        var a = Reframe(frame);
        var b = other.Reframe(frame);
        var d = a.Data;
        for (int i = 0; i < d.Length; i++) d[i] = (byte)(op(d[i] != 0 ? 1 : 0, b.Data[i] != 0 ? 1 : 0) != 0 ? 1 : 0);
        return a.Trim();
    }

    public long OverlapCount(RasterMask other)
    {
        var frame = Bounds.Intersect(other.Bounds);
        long n = 0;
        for (int y = frame.Y; y < frame.Bottom; y++)
            for (int x = frame.X; x < frame.Right; x++)
                if (Get(x, y) && other.Get(x, y)) n++;
        return n;
    }

    public (double X, double Y) Centroid()
    {
        double sx = 0, sy = 0; long n = 0;
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
                if (Data[y * Width + x] != 0) { sx += x; sy += y; n++; }
        return n == 0 ? (double.NaN, double.NaN) : (Bounds.X + sx / n + 0.5, Bounds.Y + sy / n + 0.5);
    }

    /// <summary>Number of 4-connected pixel edges between set and unset pixels (pixel-boundary perimeter).</summary>
    public long CrackPerimeter()
    {
        long p = 0;
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
            {
                if (Data[y * Width + x] == 0) continue;
                if (x == 0 || Data[y * Width + x - 1] == 0) p++;
                if (x == Width - 1 || Data[y * Width + x + 1] == 0) p++;
                if (y == 0 || Data[(y - 1) * Width + x] == 0) p++;
                if (y == Height - 1 || Data[(y + 1) * Width + x] == 0) p++;
            }
        return p;
    }

    public RasterMask Clone() => new(Bounds, (byte[])Data.Clone());

    public RasterMask Translate(int dx, int dy) =>
        new(new PixelRect(Bounds.X + dx, Bounds.Y + dy, Width, Height), (byte[])Data.Clone());

    /// <summary>Pixel-set equality (independent of bounds padding).</summary>
    public bool Equals(RasterMask? other)
    {
        if (other is null) return false;
        var a = Trim(); var b = other.Trim();
        if (a.Bounds != b.Bounds) return false;
        for (int i = 0; i < a.Data.Length; i++)
            if ((a.Data[i] != 0) != (b.Data[i] != 0)) return false;
        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as RasterMask);
    public override int GetHashCode() => HashCode.Combine(Trim().Bounds, PixelCount);
}
