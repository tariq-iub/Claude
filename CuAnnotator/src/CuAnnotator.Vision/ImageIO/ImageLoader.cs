using System.Security.Cryptography;
using OpenCvSharp;

namespace CuAnnotator.Vision.ImageIO;

public sealed class ImageInfo
{
    public required string Path { get; init; }
    public required int Width { get; init; }
    public required int Height { get; init; }
    public required string PixelFormat { get; init; }
    public required int BitDepth { get; init; }
    public required int Channels { get; init; }
    public required long FileSizeBytes { get; init; }
    public required string Sha256 { get; init; }
    public int? ExifOrientation { get; init; }
    public bool ExifOrientationApplied { get; init; }
    public string ColorProfile { get; init; } = "";
    public List<string> ConversionNotes { get; init; } = new();
}

/// <summary>A decoded image as 8-bit BGR for display and segmentation. Dispose to release native memory.</summary>
public sealed class LoadedImage : IDisposable
{
    public required Mat Bgr { get; init; }
    public required ImageInfo Info { get; init; }
    public void Dispose() => Bgr.Dispose();
}

/// <summary>
/// Image ingestion. Files are never modified.
/// <list type="bullet">
/// <item>Decoding uses <c>IMREAD_ANYDEPTH | IMREAD_ANYCOLOR</c>, which applies the EXIF orientation tag
/// (pixels are stored upright; the tag is recorded).</item>
/// <item>16-bit images are converted to 8-bit by linear scaling (v/257); 32-bit float by clamping [0,1]·255.
/// The conversion is recorded in <see cref="ImageInfo.ConversionNotes"/>.</item>
/// <item>Grayscale is replicated to 3 channels; alpha is dropped.</item>
/// <item>Embedded ICC profiles are detected and recorded but not applied: values are treated as sRGB
/// and are not quantitative colour measurements.</item>
/// </list>
/// </summary>
public static class ImageLoader
{
    public static readonly string[] SupportedExtensions = { ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp" };

    public static bool IsSupported(string path) =>
        SupportedExtensions.Contains(System.IO.Path.GetExtension(path).ToLowerInvariant());

    public static LoadedImage Load(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("Image not found.", path);
        var bytes = File.ReadAllBytes(path);
        var raw = Cv2.ImDecode(bytes, ImreadModes.AnyDepth | ImreadModes.AnyColor);
        if (raw is null || raw.Empty()) throw new InvalidDataException($"OpenCV could not decode '{path}'.");
        var notes = new List<string>();
        int depthBits = raw.Depth() switch
        {
            MatType.CV_8U or MatType.CV_8S => 8,
            MatType.CV_16U or MatType.CV_16S => 16,
            MatType.CV_32F => 32,
            _ => 64,
        };
        int channels = raw.Channels();
        string fmt = $"{depthBits}{(raw.Depth() == MatType.CV_32F ? "F" : "U")}C{channels}";

        Mat eight = raw;
        if (raw.Depth() == MatType.CV_16U)
        {
            eight = new Mat(); raw.ConvertTo(eight, MatType.CV_8U, 1.0 / 257);
            notes.Add("16-bit → 8-bit by linear scaling v/257 for display and segmentation.");
        }
        else if (raw.Depth() == MatType.CV_32F)
        {
            eight = new Mat(); raw.ConvertTo(eight, MatType.CV_8U, 255);
            notes.Add("32-bit float → 8-bit by clamping [0,1]·255.");
        }
        else if (raw.Depth() != MatType.CV_8U)
        {
            eight = new Mat(); raw.ConvertTo(eight, MatType.CV_8U);
            notes.Add($"Depth {raw.Depth()} converted to 8-bit with saturation.");
        }
        var bgr = new Mat();
        switch (channels)
        {
            case 1: Cv2.CvtColor(eight, bgr, ColorConversionCodes.GRAY2BGR); notes.Add("Grayscale replicated to 3 channels."); break;
            case 4: Cv2.CvtColor(eight, bgr, ColorConversionCodes.BGRA2BGR); notes.Add("Alpha channel dropped."); break;
            default: eight.CopyTo(bgr); break;
        }
        if (!ReferenceEquals(eight, raw)) eight.Dispose();
        raw.Dispose();

        int? orientation = ExifReader.ReadOrientation(bytes);
        var icc = DetectIcc(bytes);
        return new LoadedImage
        {
            Bgr = bgr,
            Info = new ImageInfo
            {
                Path = path, Width = bgr.Width, Height = bgr.Height, PixelFormat = fmt, BitDepth = depthBits,
                Channels = channels, FileSizeBytes = bytes.LongLength,
                Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),
                ExifOrientation = orientation,
                ExifOrientationApplied = orientation is > 1 and <= 8,
                ColorProfile = icc ? "Embedded ICC profile present (recorded, not applied; values treated as sRGB)" : "None embedded (assumed sRGB)",
                ConversionNotes = notes,
            },
        };
    }

    public static string ComputeSha256(string path)
    {
        using var s = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant();
    }

    private static bool DetectIcc(byte[] b)
    {
        // JPEG APP2 "ICC_PROFILE", PNG iCCP chunk, TIFF tag 34675 (searched in the first 64 KiB).
        var span = b.AsSpan(0, Math.Min(b.Length, 65536));
        return span.IndexOf("ICC_PROFILE"u8) >= 0 || span.IndexOf("iCCP"u8) >= 0
               || IndexOfTiffTag(span, 34675);
    }

    private static bool IndexOfTiffTag(ReadOnlySpan<byte> s, ushort tag)
    {
        if (s.Length < 8) return false;
        bool le = s[0] == 'I' && s[1] == 'I', be = s[0] == 'M' && s[1] == 'M';
        if (!le && !be) return false;
        return ExifReader.FindTag(s, 0, le, tag) is not null;
    }
}

/// <summary>Minimal EXIF/TIFF IFD0 reader for the orientation tag (0x0112).</summary>
public static class ExifReader
{
    public static int? ReadOrientation(byte[] data)
    {
        var s = data.AsSpan();
        if (s.Length > 4 && s[0] == 0xFF && s[1] == 0xD8)
        {
            int p = 2;
            while (p + 4 < s.Length && s[p] == 0xFF)
            {
                byte marker = s[p + 1];
                int len = (s[p + 2] << 8) | s[p + 3];
                if (marker == 0xE1 && p + 10 < s.Length && s.Slice(p + 4, 6).SequenceEqual("Exif\0\0"u8))
                {
                    var tiff = s.Slice(p + 10, Math.Min(len - 8, s.Length - p - 10));
                    return OrientationFromTiff(tiff);
                }
                if (marker == 0xDA) break;
                p += 2 + len;
            }
            return null;
        }
        if (s.Length > 8 && ((s[0] == 'I' && s[1] == 'I') || (s[0] == 'M' && s[1] == 'M'))) return OrientationFromTiff(s);
        return null;
    }

    private static int? OrientationFromTiff(ReadOnlySpan<byte> t)
    {
        if (t.Length < 8) return null;
        bool le = t[0] == 'I';
        return FindTag(t, 0, le, 0x0112);
    }

    internal static int? FindTag(ReadOnlySpan<byte> t, int _, bool le, ushort tag)
    {
        uint ifd = U32(t, 4, le);
        if (ifd + 2 > t.Length) return null;
        int count = U16(t, (int)ifd, le);
        for (int i = 0; i < count; i++)
        {
            int e = (int)ifd + 2 + i * 12;
            if (e + 12 > t.Length) return null;
            if (U16(t, e, le) == tag) return U16(t, e + 8, le);
        }
        return null;
    }

    private static ushort U16(ReadOnlySpan<byte> t, int o, bool le) =>
        le ? (ushort)(t[o] | (t[o + 1] << 8)) : (ushort)((t[o] << 8) | t[o + 1]);

    private static uint U32(ReadOnlySpan<byte> t, int o, bool le) =>
        le ? (uint)(t[o] | (t[o + 1] << 8) | (t[o + 2] << 16) | (t[o + 3] << 24))
           : (uint)((t[o] << 24) | (t[o + 1] << 16) | (t[o + 2] << 8) | t[o + 3]);
}
