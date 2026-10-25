using System.Runtime.InteropServices;
using CuAnnotator.Domain.Geometry;
using OpenCvSharp;

namespace CuAnnotator.Vision.Interop;

/// <summary>Conversions between OpenCV matrices and domain types. Callers own returned Mats.</summary>
public static class MatInterop
{
    public static Rect ToRect(this PixelRect r) => new(r.X, r.Y, r.Width, r.Height);
    public static PixelRect ToPixelRect(this Rect r) => new(r.X, r.Y, r.Width, r.Height);
    public static Point ToCv(this PixelPoint p) => new(p.X, p.Y);

    /// <summary>Copies a single-channel 8-bit Mat into a byte array (row-major, no padding).</summary>
    public static byte[] ToBytes(Mat m)
    {
        if (m.Type() != MatType.CV_8UC1) throw new ArgumentException($"Expected CV_8UC1, got {m.Type()}.");
        int rows = m.Rows, cols = m.Cols;
        var data = new byte[rows * cols];
        if (m.IsContinuous()) Marshal.Copy(m.Data, data, 0, data.Length);
        else
            for (int y = 0; y < rows; y++)
                Marshal.Copy(m.Ptr(y), data, y * cols, cols);
        return data;
    }

    public static float[] ToFloats(Mat m)
    {
        if (m.Depth() != MatType.CV_32F) throw new ArgumentException("Expected 32F Mat.");
        int rows = m.Rows, rowLen = m.Cols * m.Channels();
        var data = new float[rows * rowLen];
        for (int y = 0; y < rows; y++) Marshal.Copy(m.Ptr(y), data, y * rowLen, rowLen);
        return data;
    }

    public static Mat FromBytes(byte[] data, int width, int height)
    {
        var m = new Mat(height, width, MatType.CV_8UC1);
        if (data.Length > 0) Marshal.Copy(data, 0, m.Data, data.Length);
        return m;
    }

    /// <summary>Renders a mask into a 0/1 CV_8UC1 Mat covering <paramref name="frame"/>.</summary>
    public static Mat MaskToMat(RasterMask mask, PixelRect frame)
    {
        var r = mask.Reframe(frame);
        return FromBytes(r.Data, frame.Width, frame.Height);
    }

    /// <summary>Mat (any non-zero = set) over <paramref name="frame"/> → trimmed RasterMask.</summary>
    public static RasterMask MatToMask(Mat m, PixelRect frame)
    {
        var bytes = ToBytes(m);
        for (int i = 0; i < bytes.Length; i++) bytes[i] = bytes[i] != 0 ? (byte)1 : (byte)0;
        return new RasterMask(frame, bytes).Trim();
    }

    /// <summary>Paints strokes (polylines with radius) into a CV_8UC1 Mat in frame-local coordinates.</summary>
    public static void PaintStrokes(Mat target, IEnumerable<Segmentation.Stroke> strokes, PixelRect frame, byte value)
    {
        foreach (var s in strokes)
        {
            if (s.Points.Count == 0) continue;
            int thickness = Math.Max(1, s.Radius * 2 + 1);
            var pts = s.Points.Select(p => new Point(p.X - frame.X, p.Y - frame.Y)).ToArray();
            if (pts.Length == 1) Cv2.Circle(target, pts[0], Math.Max(0, s.Radius), Scalar.All(value), -1, LineTypes.Link8);
            else
                for (int i = 1; i < pts.Length; i++)
                    Cv2.Line(target, pts[i - 1], pts[i], Scalar.All(value), thickness, LineTypes.Link8);
        }
    }
}
