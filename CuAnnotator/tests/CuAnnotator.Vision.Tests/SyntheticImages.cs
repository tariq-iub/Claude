using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Vision.Tests;

/// <summary>Deterministic synthetic test images with known ground truth.</summary>
public static class SyntheticImages
{
    public static readonly Scalar Copper = new(70, 120, 190);   // BGR, warm copper
    public static readonly Scalar Oxide = new(40, 40, 150);     // dark red (Cu2O-like)
    public static readonly Scalar Patina = new(110, 160, 60);   // green (malachite-like)

    /// <summary>Background with seeded Gaussian noise, then shapes drawn with the given colour.</summary>
    public static Mat Background(int w, int h, double noiseSigma = 4, int seed = 7)
    {
        var img = new Mat(h, w, MatType.CV_8UC3, Copper);
        AddNoise(img, noiseSigma, seed);
        return img;
    }

    public static void AddNoise(Mat img, double sigma, int seed)
    {
        if (sigma <= 0) return;
        Cv2.SetTheRNG((ulong)seed);
        using var noise = new Mat(img.Size(), MatType.CV_16SC3);
        Cv2.Randn(noise, Scalar.All(0), Scalar.All(sigma));
        using var tmp = new Mat();
        img.ConvertTo(tmp, MatType.CV_16SC3);
        Cv2.Add(tmp, noise, tmp);
        tmp.ConvertTo(img, MatType.CV_8UC3);
    }

    /// <summary>Ground truth mask from drawing into a 0/1 Mat with the same calls.</summary>
    public static RasterMask Truth(int w, int h, Action<Mat> draw)
    {
        using var m = new Mat(h, w, MatType.CV_8UC1, Scalar.All(0));
        draw(m);
        return MatInterop.MatToMask(m, new PixelRect(0, 0, w, h));
    }

    public static double IoU(RasterMask a, RasterMask b)
    {
        long inter = a.OverlapCount(b);
        long union = a.PixelCount + b.PixelCount - inter;
        return union == 0 ? 1 : (double)inter / union;
    }

    /// <summary>Random blob mask (union of seeded circles/ellipses) with holes, for round-trip tests.</summary>
    public static RasterMask RandomBlobs(int w, int h, int seed, int count = 12)
    {
        var rnd = new Random(seed);
        using var m = new Mat(h, w, MatType.CV_8UC1, Scalar.All(0));
        for (int i = 0; i < count; i++)
        {
            var c = new Point(rnd.Next(w), rnd.Next(h));
            var axes = new Size(rnd.Next(1, w / 6), rnd.Next(1, h / 6));
            Cv2.Ellipse(m, c, axes, rnd.Next(180), 0, 360, Scalar.All(1), -1, LineTypes.Link8);
        }
        for (int i = 0; i < count / 2; i++)
            Cv2.Circle(m, new Point(rnd.Next(w), rnd.Next(h)), rnd.Next(1, w / 15), Scalar.All(0), -1, LineTypes.Link8);
        for (int i = 0; i < 20; i++) m.Set(rnd.Next(h), rnd.Next(w), (byte)1); // isolated pixels
        return MatInterop.MatToMask(m, new PixelRect(0, 0, w, h));
    }
}
