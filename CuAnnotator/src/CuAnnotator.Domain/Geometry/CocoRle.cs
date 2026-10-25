namespace CuAnnotator.Domain.Geometry;

/// <summary>
/// COCO run-length encoding (pycocotools <c>maskApi.c</c> semantics):
/// pixels are scanned in column-major (Fortran) order, runs alternate starting with 0-valued pixels,
/// and the compressed string form uses the LEB128-like 5-bit scheme with delta coding from run i-2.
/// Used both for COCO export and for lossless mask storage inside native project files.
/// </summary>
public static class CocoRle
{
    /// <summary>Uncompressed counts for a full-frame mask of size width × height.</summary>
    public static List<long> Encode(Func<int, int, bool> pixel, int width, int height)
    {
        var counts = new List<long>();
        bool current = false; long run = 0;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                bool v = pixel(x, y);
                if (v != current) { counts.Add(run); run = 0; current = v; }
                run++;
            }
        counts.Add(run);
        return counts;
    }

    public static List<long> Encode(RasterMask mask, int imageWidth, int imageHeight) =>
        Encode(mask.Get, imageWidth, imageHeight);

    /// <summary>Encode a mask relative to its own bounds (used for native storage).</summary>
    public static List<long> EncodeLocal(RasterMask mask) =>
        Encode((x, y) => mask.Data[y * mask.Width + x] != 0, mask.Width, mask.Height);

    public static byte[] Decode(IReadOnlyList<long> counts, int width, int height)
    {
        var data = new byte[width * height];
        long idx = 0; byte v = 0;
        long total = (long)width * height;
        foreach (var c in counts)
        {
            if (v == 1)
                for (long k = idx; k < idx + c && k < total; k++)
                {
                    int x = (int)(k / height), y = (int)(k % height);
                    data[y * width + x] = 1;
                }
            idx += c;
            v ^= 1;
        }
        if (idx != total) throw new FormatException($"RLE covers {idx} pixels but frame has {total}.");
        return data;
    }

    public static RasterMask DecodeLocal(IReadOnlyList<long> counts, PixelRect bounds) =>
        new(bounds, Decode(counts, bounds.Width, bounds.Height));

    /// <summary>pycocotools <c>rleToString</c>.</summary>
    public static string ToCompressedString(IReadOnlyList<long> counts)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < counts.Count; i++)
        {
            long x = counts[i];
            if (i > 2) x -= counts[i - 2];
            bool more = true;
            while (more)
            {
                long c = x & 0x1f;
                x >>= 5;
                more = (c & 0x10) != 0 ? x != -1 : x != 0;
                if (more) c |= 0x20;
                sb.Append((char)(c + 48));
            }
        }
        return sb.ToString();
    }

    /// <summary>pycocotools <c>rleFrString</c>.</summary>
    public static List<long> FromCompressedString(string s)
    {
        var counts = new List<long>();
        int p = 0;
        while (p < s.Length)
        {
            long x = 0; int k = 0; bool more = true;
            while (more)
            {
                long c = s[p] - 48;
                x |= (c & 0x1f) << (5 * k);
                more = (c & 0x20) != 0;
                p++; k++;
                if (!more && (c & 0x10) != 0) x |= -1L << (5 * k);
            }
            if (counts.Count > 2) x += counts[^2];
            counts.Add(x);
        }
        return counts;
    }
}
