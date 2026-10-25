using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Contours;
using CuAnnotator.Vision.Geometry;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Vision.Tests;

public class GeometryRoundTripTests
{
    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    [InlineData(6)] [InlineData(7)] [InlineData(8)] [InlineData(9)] [InlineData(10)]
    public void MaskToPolygonsToMask_IsExact(int seed)
    {
        var mask = SyntheticImages.RandomBlobs(160, 120, seed);
        var comps = ContourExtractor.Extract(mask);
        var back = MaskRasterizer.Rasterize(comps, 160, 120);
        Assert.Equal(mask.PixelCount, back.PixelCount);
        Assert.True(mask.Equals(back));
    }

    [Fact]
    public void Extract_FindsHolesAndDisconnectedComponents()
    {
        var truth = SyntheticImages.Truth(100, 100, m =>
        {
            Cv2.Rectangle(m, new Rect(10, 10, 40, 40), Scalar.All(1), -1);
            Cv2.Rectangle(m, new Rect(20, 20, 10, 10), Scalar.All(0), -1); // hole
            Cv2.Circle(m, new Point(75, 75), 10, Scalar.All(1), -1);      // second component
            Cv2.Circle(m, new Point(25, 25), 2, Scalar.All(1), -1);       // island inside the hole
        });
        var comps = ContourExtractor.Extract(truth);
        Assert.Equal(3, comps.Count);
        Assert.Single(comps[0].Holes);
        Assert.True(truth.Equals(MaskRasterizer.Rasterize(comps, 100, 100)));
    }

    [Fact]
    public void Rasterize_MaskTouchingImageBorder_IsExact()
    {
        var truth = SyntheticImages.Truth(50, 40, m => Cv2.Rectangle(m, new Rect(0, 0, 50, 10), Scalar.All(1), -1));
        Assert.True(truth.Equals(MaskRasterizer.Rasterize(ContourExtractor.Extract(truth), 50, 40)));
    }

    [Fact]
    public void Simplification_ReducesVerticesWithoutTouchingTheMask()
    {
        var truth = SyntheticImages.Truth(200, 200, m => Cv2.Circle(m, new Point(100, 100), 70, Scalar.All(1), -1));
        var exact = ContourExtractor.Extract(truth);
        var simple = ContourExtractor.Extract(truth, simplifyEpsilon: 1.5);
        Assert.True(simple[0].VertexCount < exact[0].VertexCount / 4);
        var approx = MaskRasterizer.Rasterize(simple, 200, 200);
        Assert.True(SyntheticImages.IoU(truth, approx) > 0.97);
    }

    [Fact]
    public void Split_ByCutLine_ProducesTwoPartsCoveringEveryPixel()
    {
        var truth = SyntheticImages.Truth(100, 60, m => Cv2.Rectangle(m, new Rect(10, 10, 80, 40), Scalar.All(1), -1));
        var parts = SegmentationEngine.Split(truth, new[] { new PixelPoint(50, 0), new PixelPoint(50, 59) });
        Assert.Equal(2, parts.Count);
        Assert.Equal(truth.PixelCount, parts.Sum(p => p.PixelCount));
        Assert.Equal(0, parts[0].OverlapCount(parts[1]));
    }

    [Fact]
    public void FillHoles_FillsInteriorOnly()
    {
        var ring = SyntheticImages.Truth(60, 60, m =>
        {
            Cv2.Circle(m, new Point(30, 30), 20, Scalar.All(1), -1);
            Cv2.Circle(m, new Point(30, 30), 8, Scalar.All(0), -1);
        });
        var filled = SegmentationEngine.FillHoles(ring);
        var disc = SyntheticImages.Truth(60, 60, m => Cv2.Circle(m, new Point(30, 30), 20, Scalar.All(1), -1));
        Assert.True(filled.Equals(disc));
    }

    [Fact]
    public void BoundarySnapper_MovesPathOntoNearbyEdge()
    {
        using var img = new Mat(80, 80, MatType.CV_8UC3, Scalar.All(40));
        Cv2.Rectangle(img, new Rect(40, 0, 40, 80), Scalar.All(220), -1); // vertical edge at x = 39.5
        var path = new[] { new PixelPoint(34, 10), new PixelPoint(34, 70) };
        var snapped = BoundarySnapper.Snap(img, path, searchRadius: 8);
        Assert.All(snapped, p => Assert.InRange(p.X, 38, 41));
    }
}
