using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.ColorAnalysis;

public sealed class RegionStatistics
{
    public long Area { get; init; }
    public double Perimeter { get; init; }
    /// <summary>4πA/P² (1 for a disc).</summary>
    public double Circularity { get; init; }
    /// <summary>P²/A.</summary>
    public double Compactness { get; init; }
    /// <summary>Long/short side of the minimum-area rotated rectangle.</summary>
    public double AspectRatio { get; init; }
    public double[] MeanRgb { get; init; } = new double[3];
    public double[] MedianRgb { get; init; } = new double[3];
    public double[] MeanLab { get; init; } = new double[3];
    public double[] MedianLab { get; init; } = new double[3];
    public double[] StdLab { get; init; } = new double[3];
    public double[] MinLab { get; init; } = new double[3];
    public double[] MaxLab { get; init; } = new double[3];
    /// <summary>L* histogram, 20 bins over [0,100].</summary>
    public int[] HistogramL { get; init; } = new int[20];
    /// <summary>a* histogram, 32 bins over [−128,128].</summary>
    public int[] HistogramA { get; init; } = new int[32];
    /// <summary>b* histogram, 32 bins over [−128,128].</summary>
    public int[] HistogramB { get; init; } = new int[32];
    /// <summary>Fraction of region pixels on Canny edges (texture proxy).</summary>
    public double EdgeDensity { get; init; }
    /// <summary>Mean Sobel gradient magnitude of L* inside the region.</summary>
    public double MeanGradient { get; init; }

    /// <summary>Descriptor for class-assisted suggestions (order documented in 05_Segmentation_Algorithms.md).</summary>
    public double[] ToDescriptor() => new[]
    {
        MeanLab[0], MeanLab[1], MeanLab[2], MedianLab[0], MedianLab[1], MedianLab[2],
        StdLab[0], StdLab[1], StdLab[2], EdgeDensity, MeanGradient,
    };

    public static readonly string[] DescriptorNames =
        { "meanL", "meanA", "meanB", "medL", "medA", "medB", "stdL", "stdA", "stdB", "edgeDensity", "meanGradient" };

    public static RegionStatistics Compute(Mat bgrImage, RasterMask mask)
    {
        var frame = mask.Bounds.ClampTo(bgrImage.Width, bgrImage.Height);
        var sel = mask.Reframe(frame).Data;
        long n = sel.Count(b => b != 0);
        if (n == 0) return new RegionStatistics();

        using var crop = new Mat(bgrImage, frame.ToRect());
        var lab = LabImage.FromBgr(bgrImage, frame);
        var bgr = new byte[frame.Width * frame.Height * 3];
        for (int y = 0; y < frame.Height; y++)
            System.Runtime.InteropServices.Marshal.Copy(crop.Ptr(y), bgr, y * frame.Width * 3, frame.Width * 3);

        var ch = Enumerable.Range(0, 6).Select(_ => new List<double>((int)n)).ToArray(); // R,G,B,L,a,b
        var hl = new int[20]; var ha = new int[32]; var hb = new int[32];
        for (int i = 0; i < sel.Length; i++)
        {
            if (sel[i] == 0) continue;
            ch[0].Add(bgr[i * 3 + 2]); ch[1].Add(bgr[i * 3 + 1]); ch[2].Add(bgr[i * 3]);
            double L = lab.Data[i * 3], A = lab.Data[i * 3 + 1], B = lab.Data[i * 3 + 2];
            ch[3].Add(L); ch[4].Add(A); ch[5].Add(B);
            hl[Math.Clamp((int)(L / 5), 0, 19)]++;
            ha[Math.Clamp((int)((A + 128) / 8), 0, 31)]++;
            hb[Math.Clamp((int)((B + 128) / 8), 0, 31)]++;
        }

        using var gray = new Mat();
        Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);
        using var edges = new Mat();
        Cv2.Canny(gray, edges, 40, 120);
        var eb = MatInterop.ToBytes(edges);
        using var gx = new Mat(); using var gy = new Mat(); using var mag = new Mat();
        using var gf = new Mat();
        gray.ConvertTo(gf, MatType.CV_32F);
        Cv2.Sobel(gf, gx, MatType.CV_32F, 1, 0); Cv2.Sobel(gf, gy, MatType.CV_32F, 0, 1);
        Cv2.Magnitude(gx, gy, mag);
        var mg = MatInterop.ToFloats(mag);
        long edgeCount = 0; double gradSum = 0;
        for (int i = 0; i < sel.Length; i++)
            if (sel[i] != 0) { if (eb[i] != 0) edgeCount++; gradSum += mg[i]; }

        var comps = Contours.ContourExtractor.Extract(mask);
        double perim = comps.Sum(c => PolygonMath.Perimeter(c.Outer) + c.Holes.Sum(h => PolygonMath.Perimeter(h)));
        double aspect = 1;
        var allPts = comps.SelectMany(c => c.Outer).Select(p => new Point2f(p.X, p.Y)).ToArray();
        if (allPts.Length >= 3)
        {
            var rr = Cv2.MinAreaRect(allPts);
            float a = Math.Max(rr.Size.Width, rr.Size.Height) + 1, b = Math.Min(rr.Size.Width, rr.Size.Height) + 1;
            aspect = a / b;
        }

        return new RegionStatistics
        {
            Area = n,
            Perimeter = perim,
            Circularity = perim > 0 ? 4 * Math.PI * n / (perim * perim) : 0,
            Compactness = n > 0 ? perim * perim / n : 0,
            AspectRatio = aspect,
            MeanRgb = ch[..3].Select(c => c.Average()).ToArray(),
            MedianRgb = ch[..3].Select(Median).ToArray(),
            MeanLab = ch[3..].Select(c => c.Average()).ToArray(),
            MedianLab = ch[3..].Select(Median).ToArray(),
            StdLab = ch[3..].Select(Std).ToArray(),
            MinLab = ch[3..].Select(c => c.Min()).ToArray(),
            MaxLab = ch[3..].Select(c => c.Max()).ToArray(),
            HistogramL = hl, HistogramA = ha, HistogramB = hb,
            EdgeDensity = (double)edgeCount / n,
            MeanGradient = gradSum / n,
        };
    }

    private static double Median(List<double> v)
    {
        var s = v.OrderBy(x => x).ToList();
        int n = s.Count;
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    }

    private static double Std(List<double> v)
    {
        double m = v.Average();
        return Math.Sqrt(v.Sum(x => (x - m) * (x - m)) / v.Count);
    }
}
