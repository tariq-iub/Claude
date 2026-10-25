using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace CuAnnotator.Vision.ColorAnalysis;

/// <summary>
/// Float CIELAB view of an image region (L* ∈ [0,100], a*, b* unscaled). Interleaved L,a,b per pixel, row-major.
/// </summary>
public sealed class LabImage
{
    public PixelRect Frame { get; }
    public float[] Data { get; }
    public int Width => Frame.Width;
    public int Height => Frame.Height;

    private LabImage(PixelRect frame, float[] data) { Frame = frame; Data = data; }

    private static readonly double[] LinearLut = Enumerable.Range(0, 256).Select(v => ColorScience.SrgbToLinear(v / 255.0)).ToArray();

    /// <summary>
    /// Exact sRGB(D65) → CIELAB for 8-bit BGR input using the same formula as
    /// <see cref="ColorScience.RgbToLab(double,double,double)"/> (linearisation via a 256-entry LUT).
    /// OpenCV's float <c>BGR2Lab</c> approximates the sRGB gamma curve by spline interpolation
    /// (≈0.2 ΔE76 deviation), so it is not used for quantitative work.
    /// </summary>
    public static LabImage FromBgr(Mat bgrImage, PixelRect frame)
    {
        using var crop = new Mat(bgrImage, frame.ToRect());
        int w = frame.Width, h = frame.Height;
        var bgr = new byte[w * h * 3];
        for (int y = 0; y < h; y++) Marshal.Copy(crop.Ptr(y), bgr, y * w * 3, w * 3);
        var data = new float[w * h * 3];
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 3;
                double rl = LinearLut[bgr[i + 2]], gl = LinearLut[bgr[i + 1]], bl = LinearLut[bgr[i]];
                double X = (0.412453 * rl + 0.357580 * gl + 0.180423 * bl) / ColorScience.Xn;
                double Y = 0.212671 * rl + 0.715160 * gl + 0.072169 * bl;
                double Z = (0.019334 * rl + 0.119193 * gl + 0.950227 * bl) / ColorScience.Zn;
                double fx = F(X), fy = F(Y), fz = F(Z);
                data[i] = (float)(Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y);
                data[i + 1] = (float)(500 * (fx - fy));
                data[i + 2] = (float)(200 * (fy - fz));
            }
        });
        return new LabImage(frame, data);
    }

    private static double F(double t) => t > 0.008856 ? Math.Cbrt(t) : 7.787 * t + 16.0 / 116;

    public LabColor At(int localX, int localY)
    {
        int i = (localY * Width + localX) * 3;
        return new LabColor(Data[i], Data[i + 1], Data[i + 2]);
    }

    /// <summary>Per-channel median of the pixels selected by a 0/1 selection over this frame (robust representative colour).</summary>
    public LabColor? Median(byte[] selection)
    {
        var l = new List<float>(); var a = new List<float>(); var b = new List<float>();
        for (int i = 0; i < selection.Length; i++)
        {
            if (selection[i] == 0) continue;
            l.Add(Data[i * 3]); a.Add(Data[i * 3 + 1]); b.Add(Data[i * 3 + 2]);
        }
        if (l.Count == 0) return null;
        return new LabColor(Med(l), Med(a), Med(b));
    }

    private static double Med(List<float> v)
    {
        v.Sort();
        int n = v.Count;
        return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
    }

    /// <summary>ΔE of every pixel to <paramref name="reference"/>, computed in parallel over rows.</summary>
    public float[] DistanceMap(LabColor reference, ColorDifferenceMetric metric)
    {
        var d = new float[Width * Height];
        int w = Width;
        Parallel.For(0, Height, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                var c = new LabColor(Data[i * 3], Data[i * 3 + 1], Data[i * 3 + 2]);
                d[i] = (float)ColorScience.Difference(c, reference, metric);
            }
        });
        return d;
    }
}
