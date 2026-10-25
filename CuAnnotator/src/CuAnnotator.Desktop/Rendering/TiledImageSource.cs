using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using Rect = System.Windows.Rect;

namespace CuAnnotator.Desktop.Rendering;

/// <summary>
/// Display-only multi-resolution tile pyramid over the full-resolution BGR image. Level k is the image
/// downsampled by 2^k (INTER_AREA). Tiles of 512×512 are converted to frozen WPF bitmaps on demand and
/// kept in an LRU cache, so even very large images never require a single full-size WPF bitmap and only
/// visible tiles at the zoom-appropriate level are materialised. Annotation coordinates never depend on
/// this pyramid (it is used for drawing only).
/// </summary>
public sealed class TiledImageSource : IDisposable
{
    public const int TileSize = 512;
    private const int CacheCapacity = 160; // ≈ 120 MB worst case at 24 bpp

    private readonly Mat _level0; // not owned
    private readonly List<Mat?> _levels = new();
    private readonly Dictionary<(int L, int X, int Y), LinkedListNode<((int, int, int) Key, BitmapSource Bmp)>> _map = new();
    private readonly LinkedList<((int, int, int) Key, BitmapSource Bmp)> _lru = new();

    public int Width { get; }
    public int Height { get; }
    public int MaxLevel { get; }

    public TiledImageSource(Mat bgr)
    {
        _level0 = bgr;
        Width = bgr.Width; Height = bgr.Height;
        int lv = 0;
        while (Math.Max(Width >> lv, Height >> lv) > TileSize && lv < 12) lv++;
        MaxLevel = lv;
        for (int i = 0; i <= MaxLevel; i++) _levels.Add(null);
        _levels[0] = _level0;
    }

    public Mat Level(int k)
    {
        if (_levels[k] is { } m) return m;
        var prev = Level(k - 1);
        var next = new Mat();
        Cv2.Resize(prev, next, new OpenCvSharp.Size(Math.Max(1, prev.Width / 2), Math.Max(1, prev.Height / 2)), 0, 0, InterpolationFlags.Area);
        _levels[k] = next;
        return next;
    }

    /// <summary>Pyramid level whose resolution best matches the display scale (never coarser than needed).</summary>
    public int LevelFor(double scale) => Math.Clamp((int)Math.Floor(Math.Log2(1.0 / Math.Max(scale, 1e-9))), 0, MaxLevel);

    /// <summary>Visible tiles at a level: tile bitmap plus its rectangle in level-0 image coordinates.</summary>
    public IEnumerable<(BitmapSource Bitmap, Rect ImageRect)> VisibleTiles(int level, Rect visibleImageRect)
    {
        var m = Level(level);
        double f = (double)Width / m.Width;          // level → image scale (≈ 2^level)
        double fy = (double)Height / m.Height;
        int tx0 = Math.Max(0, (int)Math.Floor(visibleImageRect.Left / f / TileSize));
        int ty0 = Math.Max(0, (int)Math.Floor(visibleImageRect.Top / fy / TileSize));
        int tx1 = Math.Min((m.Width - 1) / TileSize, (int)Math.Floor(visibleImageRect.Right / f / TileSize));
        int ty1 = Math.Min((m.Height - 1) / TileSize, (int)Math.Floor(visibleImageRect.Bottom / fy / TileSize));
        for (int ty = ty0; ty <= ty1; ty++)
            for (int tx = tx0; tx <= tx1; tx++)
            {
                var r = new Rect(tx * TileSize, ty * TileSize, Math.Min(TileSize, m.Width - tx * TileSize), Math.Min(TileSize, m.Height - ty * TileSize));
                yield return (Tile(level, tx, ty, m, r), new Rect(r.X * f, r.Y * fy, r.Width * f, r.Height * fy));
            }
    }

    private BitmapSource Tile(int level, int tx, int ty, Mat m, Rect r)
    {
        var key = (level, tx, ty);
        if (_map.TryGetValue(key, out var node))
        {
            _lru.Remove(node); _lru.AddFirst(node);
            return node.Value.Bmp;
        }
        using var sub = new Mat(m, new OpenCvSharp.Rect((int)r.X, (int)r.Y, (int)r.Width, (int)r.Height));
        var bmp = ToBitmap(sub);
        var n = _lru.AddFirst((key, bmp));
        _map[key] = n;
        while (_lru.Count > CacheCapacity)
        {
            var last = _lru.Last!;
            _map.Remove(last.Value.Key);
            _lru.RemoveLast();
        }
        return bmp;
    }

    public static BitmapSource ToBitmap(Mat bgr)
    {
        using var c = bgr.Clone(); // continuous copy (tiles are sub-matrices)
        var bmp = BitmapSource.Create(c.Width, c.Height, 96, 96, PixelFormats.Bgr24, null, c.Data, (int)(c.Step() * c.Height), (int)c.Step());
        bmp.Freeze();
        return bmp;
    }

    /// <summary>Small preview (thumbnails, minimap).</summary>
    public BitmapSource Thumbnail(int maxSize)
    {
        double s = Math.Min(1.0, (double)maxSize / Math.Max(Width, Height));
        int lv = LevelFor(s);
        var m = Level(lv);
        using var t = new Mat();
        Cv2.Resize(m, t, new OpenCvSharp.Size(Math.Max(1, (int)(Width * s)), Math.Max(1, (int)(Height * s))), 0, 0, InterpolationFlags.Area);
        return ToBitmap(t);
    }

    public void Dispose()
    {
        for (int i = 1; i < _levels.Count; i++) _levels[i]?.Dispose();
        _lru.Clear(); _map.Clear();
    }
}
