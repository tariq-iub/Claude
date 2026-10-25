using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Vision.Tests;

public class SegmentationTests
{
    private static readonly SegmentationEngine Engine = new();

    private static (Mat Image, RasterMask Truth) IrregularScene(int w = 200, int h = 160, double noise = 4)
    {
        Action<Mat, Scalar> draw = (m, c) =>
        {
            var poly = new[] { new Point(60, 40), new Point(110, 30), new Point(140, 70), new Point(115, 80), new Point(130, 115), new Point(70, 120), new Point(85, 85) };
            Cv2.FillPoly(m, new[] { poly }, c, LineTypes.Link8);
        };
        var img = new Mat(h, w, MatType.CV_8UC3, SyntheticImages.Copper);
        draw(img, SyntheticImages.Oxide);
        SyntheticImages.AddNoise(img, noise, 11);
        var truth = SyntheticImages.Truth(w, h, m => draw(m, Scalar.All(1)));
        return (img, truth);
    }

    public static TheoryData<SegmentationAlgorithmKind> AllKinds => new()
    {
        SegmentationAlgorithmKind.GrabCut, SegmentationAlgorithmKind.LabColor, SegmentationAlgorithmKind.Watershed,
        SegmentationAlgorithmKind.ActiveContour, SegmentationAlgorithmKind.Hybrid,
    };

    [Theory]
    [MemberData(nameof(AllKinds))]
    public async Task Algorithm_RecoversConcaveIrregularRegion(SegmentationAlgorithmKind kind)
    {
        var (img, truth) = IrregularScene();
        using var _ = img;
        var p = new SegmentationParameters { Algorithm = kind };
        var result = await Engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = new PixelRect(45, 20, 110, 115), Parameters = p }, default);
        double iou = SyntheticImages.IoU(truth, result.Mask);
        Assert.True(iou > 0.90, $"{kind}: IoU {iou:0.000}");
        Assert.NotEmpty(result.Components);
        Assert.Equal(kind.ToString(), result.ParametersUsed["Algorithm"]);
        Assert.True(new PixelRect(45, 20, 110, 115).Contains(result.Mask.Bounds.X, result.Mask.Bounds.Y));
    }

    [Fact]
    public async Task GrabCut_IsDeterministicWithFixedSeed()
    {
        var (img, _) = IrregularScene(noise: 12);
        using var _i = img;
        var req = new SegmentationRequest { Image = img, Roi = new PixelRect(45, 20, 110, 115) };
        var a = await Engine.SegmentAsync(req, default);
        var b = await Engine.SegmentAsync(req, default);
        Assert.True(a.Mask.Equals(b.Mask));
    }

    [Fact]
    public async Task GrabCut_ForegroundAndBackgroundSeeds_AreHardConstraints()
    {
        // Two similar blobs inside one ROI: the user keeps the left one and rejects the right one.
        using var img = SyntheticImages.Background(200, 120);
        Cv2.Circle(img, new Point(60, 60), 25, SyntheticImages.Oxide, -1);
        Cv2.Circle(img, new Point(140, 60), 25, SyntheticImages.Oxide, -1);
        var roi = new PixelRect(20, 20, 165, 85);
        var first = await Engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = roi }, default);
        Assert.True(first.Mask.Get(140, 60) && first.Mask.Get(60, 60));

        var refined = await Engine.SegmentAsync(new SegmentationRequest
        {
            Image = img, Roi = roi, PreviousGrabCutState = first.GrabCutState,
            PositiveStrokes = new[] { Stroke.Point(new PixelPoint(60, 60), 3) },
            NegativeStrokes = new[] { new Stroke(new[] { new PixelPoint(125, 60), new PixelPoint(155, 60) }, 4) },
        }, default);
        Assert.True(refined.Mask.Get(60, 60));
        Assert.False(refined.Mask.Get(140, 60));
        Assert.Equal("True", refined.Diagnostics["grabcut.reusedPreviousState"]);
    }

    [Fact]
    public async Task LabColor_WithoutConnectivity_SelectsDisconnectedComponents_WithConnectivity_OnlySeeded()
    {
        using var img = SyntheticImages.Background(200, 100);
        Cv2.Circle(img, new Point(40, 50), 15, SyntheticImages.Patina, -1);
        Cv2.Circle(img, new Point(150, 50), 15, SyntheticImages.Patina, -1);
        var p = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.LabColor };
        var all = await Engine.ColorSelectAsync(img, new PixelRect(0, 0, 200, 100), new[] { Stroke.Point(new PixelPoint(40, 50), 2) }, p, default);
        Assert.Equal(2, all.Components.Count);
        var wand = await Engine.MagicWandAsync(img, new PixelPoint(40, 50), p, 300, default);
        Assert.Single(wand.Components);
        Assert.True(wand.Mask.Get(40, 50) && !wand.Mask.Get(150, 50));
    }

    [Fact]
    public async Task PreserveHoles_KeepsInteriorHole()
    {
        using var img = SyntheticImages.Background(120, 120);
        Cv2.Circle(img, new Point(60, 60), 40, SyntheticImages.Oxide, -1);
        Cv2.Circle(img, new Point(60, 60), 12, SyntheticImages.Copper, -1);
        var p = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.LabColor, RequireConnectivity = true };
        var r = await Engine.SegmentAsync(new SegmentationRequest
        {
            Image = img, Roi = new PixelRect(10, 10, 100, 100), Parameters = p,
            PositiveStrokes = new[] { Stroke.Point(new PixelPoint(60, 30), 2) },
        }, default);
        Assert.False(r.Mask.Get(60, 60));
        Assert.Single(r.Components[0].Holes);

        p.PreserveHoles = false;
        var filled = await Engine.SegmentAsync(new SegmentationRequest
        {
            Image = img, Roi = new PixelRect(10, 10, 100, 100), Parameters = p,
            PositiveStrokes = new[] { Stroke.Point(new PixelPoint(60, 30), 2) },
        }, default);
        Assert.True(filled.Mask.Get(60, 60));
    }

    [Fact]
    public async Task MinComponentArea_ZeroDisablesFiltering_KeepsMicroStructures()
    {
        using var img = SyntheticImages.Background(100, 100, noiseSigma: 0);
        for (int i = 0; i < 5; i++) img.Set(20 + i * 15, 50, new Vec3b(110, 160, 60)); // single-pixel patina specks
        var p = new SegmentationParameters
        {
            Algorithm = SegmentationAlgorithmKind.LabColor, RequireConnectivity = false,
            MorphologicalCleanup = false, MinComponentArea = 0,
        };
        var r = await Engine.ColorSelectAsync(img, new PixelRect(0, 0, 100, 100), new[] { Stroke.Point(new PixelPoint(50, 20), 0) }, p, default);
        Assert.Equal(5, r.Mask.PixelCount);
        p.MinComponentArea = 5;
        var filtered = await Engine.ColorSelectAsync(img, new PixelRect(0, 0, 100, 100), new[] { Stroke.Point(new PixelPoint(50, 20), 0) }, p, default);
        // Only the speck under the sample point survives: seeds are hard constraints re-imposed after filtering.
        Assert.Equal(1, filtered.Mask.PixelCount);
        Assert.True(filtered.Mask.Get(50, 20));
    }

    [Fact]
    public async Task Hybrid_ReportsIntermediateStages()
    {
        var (img, _) = IrregularScene();
        using var _i = img;
        var r = await Engine.SegmentAsync(new SegmentationRequest
        {
            Image = img, Roi = new PixelRect(45, 20, 110, 115), Parameters = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.Hybrid },
        }, default);
        Assert.Equal(new[] { "1-grabcut", "2-cielab", "3-edge", "4-postprocessed" }, r.Stages.Select(s => s.Name));
    }

    [Fact]
    public async Task OutOfBoundsRoi_IsClippedToImage()
    {
        var (img, truth) = IrregularScene();
        using var _i = img;
        var r = await Engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = new PixelRect(45, 20, 500, 500) }, default);
        Assert.True(r.Mask.Bounds.Right <= img.Width && r.Mask.Bounds.Bottom <= img.Height);
    }

    [Fact]
    public async Task RoiOutsideImage_Throws()
    {
        using var img = SyntheticImages.Background(50, 50);
        await Assert.ThrowsAsync<SegmentationException>(() =>
            Engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = new PixelRect(100, 100, 20, 20) }, default));
    }

    [Fact]
    public async Task EmptyRegion_ProducesEmptyOrSmallMaskWithoutCrashing()
    {
        using var img = SyntheticImages.Background(80, 80, noiseSigma: 0);
        var r = await Engine.SegmentAsync(new SegmentationRequest
        {
            Image = img, Roi = new PixelRect(10, 10, 60, 60),
            Parameters = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.Watershed },
        }, default);
        Assert.NotNull(r.Mask);
    }

    [Fact]
    public async Task Cancellation_IsHonoured()
    {
        using var img = SyntheticImages.Background(400, 400);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            Engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = new PixelRect(10, 10, 300, 300) }, cts.Token));
    }

    [Fact]
    public void Parameters_RoundTripThroughDictionary()
    {
        var p = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.Hybrid, ColorThreshold = 7.25, MinComponentArea = 0, Quality = ProcessingQuality.Precise };
        var back = SegmentationParameters.FromDictionary(p.ToDictionary());
        Assert.Equal(p.ToDictionary(), back.ToDictionary());
    }
}
