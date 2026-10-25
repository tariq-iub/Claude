using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using OpenCvSharp;

namespace CuAnnotator.Vision.Tests;

public class ColorScienceTests
{
    // Sharma, Wu & Dalal (2005), Table 1: all 34 test pairs (L1,a1,b1, L2,a2,b2, ΔE00).
    public static TheoryData<double, double, double, double, double, double, double> SharmaPairs => new()
    {
        { 50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425 },
        { 50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615 },
        { 50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412 },
        { 50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000 },
        { 50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000 },
        { 50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000 },
        { 50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669 },
        { 50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669 },
        { 50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792 },
        { 50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792 },
        { 50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195 },
        { 50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195 },
        { 50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045 },
        { 50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045 },
        { 50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461 },
        { 50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065 },
        { 50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492 },
        { 50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977 },
        { 50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030 },
        { 50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535 },
        { 50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000 },
        { 50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000 },
        { 50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000 },
        { 50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000 },
        { 60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644 },
        { 63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630 },
        { 61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731 },
        { 35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645 },
        { 22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373 },
        { 36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146 },
        { 90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441 },
        { 90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381 },
        { 6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377 },
        { 2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082 },
    };

    [Theory]
    [MemberData(nameof(SharmaPairs))]
    public void Ciede2000_MatchesPublishedReferencePairs(double l1, double a1, double b1, double l2, double a2, double b2, double expected)
    {
        double d = ColorScience.DeltaE2000(new LabColor(l1, a1, b1), new LabColor(l2, a2, b2));
        Assert.Equal(expected, d, 4);
        // Symmetry.
        Assert.Equal(d, ColorScience.DeltaE2000(new LabColor(l2, a2, b2), new LabColor(l1, a1, b1)), 10);
    }

    [Theory]
    [InlineData(255, 255, 255, 100.0, 0.0, 0.0)]
    [InlineData(0, 0, 0, 0.0, 0.0, 0.0)]
    [InlineData(255, 0, 0, 53.24, 80.09, 67.20)]
    [InlineData(0, 255, 0, 87.73, -86.18, 83.18)]
    [InlineData(0, 0, 255, 32.30, 79.19, -107.86)]
    public void RgbToLab_MatchesStandardSrgbValues(byte r, byte g, byte b, double L, double A, double B)
    {
        var lab = ColorScience.RgbToLab(r, g, b);
        Assert.InRange(lab.L, L - 0.1, L + 0.1);
        Assert.InRange(lab.A, A - 0.3, A + 0.3);
        Assert.InRange(lab.B, B - 0.3, B + 0.3);
    }

    [Fact]
    public void ReferenceLab_AgreesWithOpenCvFloatConversion()
    {
        var rnd = new Random(3);
        using var img = new Mat(16, 16, MatType.CV_8UC3);
        var bytes = new byte[16 * 16 * 3];
        rnd.NextBytes(bytes);
        System.Runtime.InteropServices.Marshal.Copy(bytes, 0, img.Data, bytes.Length);
        var lab = LabImage.FromBgr(img, new PixelRect(0, 0, 16, 16));
        for (int i = 0; i < 256; i++)
        {
            var reference = ColorScience.RgbToLab(bytes[i * 3 + 2], bytes[i * 3 + 1], bytes[i * 3]);
            var cv = lab.At(i % 16, i / 16);
            Assert.True(ColorScience.DeltaE76(reference, cv) < 1e-3, $"pixel {i}: {reference} vs {cv}");
        }
    }

    [Fact]
    public void OpenCvFloatLab_DeviatesSlightlyFromExactFormula_DocumentedBound()
    {
        // Documents why CuAnnotator does not use OpenCV's float BGR2Lab for quantitative work.
        var rnd = new Random(5);
        double worst = 0;
        for (int k = 0; k < 500; k++)
        {
            byte r = (byte)rnd.Next(256), g = (byte)rnd.Next(256), b = (byte)rnd.Next(256);
            using var px = new Mat(1, 1, MatType.CV_32FC3, new Scalar(b / 255.0, g / 255.0, r / 255.0));
            using var lab = new Mat();
            Cv2.CvtColor(px, lab, ColorConversionCodes.BGR2Lab);
            var v = lab.Get<Vec3f>(0, 0);
            worst = Math.Max(worst, ColorScience.DeltaE76(ColorScience.RgbToLab(r, g, b), new LabColor(v.Item0, v.Item1, v.Item2)));
        }
        Assert.InRange(worst, 0, 1.0);
    }

    [Fact]
    public void OpenCv8BitLabScaling_IsDecodedCorrectly()
    {
        using var px = new Mat(1, 1, MatType.CV_8UC3, new Scalar(30, 140, 200));
        using var lab8 = new Mat();
        Cv2.CvtColor(px, lab8, ColorConversionCodes.BGR2Lab);
        var v = lab8.Get<Vec3b>(0, 0);
        var decoded = ColorScience.FromOpenCv8Bit(v.Item0, v.Item1, v.Item2);
        var reference = ColorScience.RgbToLab(200, 140, 30);
        Assert.True(ColorScience.DeltaE76(decoded, reference) < 1.5); // 8-bit quantisation
    }

    [Theory]
    [InlineData(255, 0, 0, 0, 1, 1)]
    [InlineData(0, 255, 0, 120, 1, 1)]
    [InlineData(0, 0, 255, 240, 1, 1)]
    [InlineData(128, 128, 128, 0, 0, 0.50196)]
    public void RgbToHsv_IsCorrect(byte r, byte g, byte b, double h, double s, double v)
    {
        var hsv = ColorScience.RgbToHsv(r, g, b);
        Assert.Equal(h, hsv.H, 3);
        Assert.Equal(s, hsv.S, 3);
        Assert.Equal(v, hsv.V, 3);
    }
}
