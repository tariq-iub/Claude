using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.ImageIO;

/// <summary>Lossless single-channel PNG I/O for label maps. Never interpolated, never JPEG.</summary>
public static class MaskIO
{
    public static void WriteLabelPng(string path, byte[] labels, int width, int height)
    {
        using var m = MatInterop.FromBytes(labels, width, height);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        if (!Cv2.ImWrite(path, m, new ImageEncodingParam(ImwriteFlags.PngCompression, 6)))
            throw new IOException($"Failed to write PNG '{path}'.");
    }

    /// <summary>Reads a single-channel 8-bit PNG exactly as stored (no colour conversion).</summary>
    public static (byte[] Labels, int Width, int Height) ReadLabelPng(string path)
    {
        using var m = Cv2.ImRead(path, ImreadModes.Unchanged);
        if (m.Empty()) throw new InvalidDataException($"Cannot read '{path}'.");
        if (m.Type() != MatType.CV_8UC1) throw new InvalidDataException($"'{path}' is {m.Type()}, expected 8-bit single channel.");
        return (MatInterop.ToBytes(m), m.Width, m.Height);
    }

    /// <summary>Writes an RGB visualisation of a label map using the given palette (display only).</summary>
    public static void WriteColorizedPng(string path, byte[] labels, int width, int height, IReadOnlyDictionary<int, (byte R, byte G, byte B)> palette)
    {
        using var m = new Mat(height, width, MatType.CV_8UC3);
        var buf = new byte[width * height * 3];
        for (int i = 0; i < labels.Length; i++)
        {
            var c = palette.TryGetValue(labels[i], out var p) ? p : ((byte)0, (byte)0, (byte)0);
            buf[i * 3] = c.Item3; buf[i * 3 + 1] = c.Item2; buf[i * 3 + 2] = c.Item1;
        }
        System.Runtime.InteropServices.Marshal.Copy(buf, 0, m.Data, buf.Length);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        Cv2.ImWrite(path, m);
    }

    public static void WriteImagePng(string path, Mat bgr)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        if (!Cv2.ImWrite(path, bgr)) throw new IOException($"Failed to write '{path}'.");
    }
}
