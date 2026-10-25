using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Domain.Validation;

namespace CuAnnotator.Domain.Tests;

public class ViewTransformTests
{
    [Theory]
    [InlineData(0.25, 13.0, -7.0)]
    [InlineData(1.0, 0.0, 0.0)]
    [InlineData(3.7, 101.5, 44.25)]
    [InlineData(16, -900, -300)]
    public void ViewToImage_IsInverseOfImageToView(double scale, double ox, double oy)
    {
        var t = new ViewTransform(scale, ox, oy);
        var p = new ImagePoint(1234.5, 987.25);
        var (vx, vy) = t.ImageToView(p);
        var back = t.ViewToImage(vx, vy);
        Assert.Equal(p.X, back.X, 9);
        Assert.Equal(p.Y, back.Y, 9);
    }

    [Fact]
    public void ZoomAt_KeepsImagePointUnderCursorFixed()
    {
        var t = new ViewTransform(0.25, 10, 20);
        var anchor = t.ViewToImage(300, 200);
        var z = t.ZoomAt(2.5, 300, 200);
        var after = z.ViewToImage(300, 200);
        Assert.Equal(anchor.X, after.X, 9);
        Assert.Equal(anchor.Y, after.Y, 9);
        Assert.Equal(0.625, z.Scale, 12);
    }

    [Fact]
    public void PolygonDrawnAt25PercentMapsToExactImagePixels()
    {
        // A click at the centre of the on-screen footprint of pixel (401, 133) at 25 % zoom.
        var t = new ViewTransform(0.25, 50, 30);
        var (vx, vy) = t.ImageToView(new ImagePoint(401.5, 133.5));
        Assert.Equal(new PixelPoint(401, 133), t.ViewToPixel(vx, vy));
        // Pan and zoom do not change the stored coordinate of the same image location.
        var t2 = t.Pan(-333, 87).ZoomAt(7, 100, 100);
        var (vx2, vy2) = t2.ImageToView(new ImagePoint(401.5, 133.5));
        Assert.Equal(new PixelPoint(401, 133), t2.ViewToPixel(vx2, vy2));
    }

    [Fact]
    public void PixelFloorRule_HandlesNegativeAndEdgeCoordinates()
    {
        Assert.Equal(new PixelPoint(-1, -1), new ImagePoint(-0.01, -0.99).ToPixel());
        Assert.Equal(new PixelPoint(0, 0), new ImagePoint(0, 0.999).ToPixel());
        Assert.Equal(new PixelPoint(5, 5), new ImagePoint(5.0, 5.5).ToPixel());
    }

    [Fact]
    public void Fit_ShowsWholeImageCentred()
    {
        var t = ViewTransform.Fit(4000, 3000, 816, 616, margin: 8);
        Assert.Equal(0.2, t.Scale, 9);
        var (x0, y0) = t.ImageToView(new ImagePoint(0, 0));
        var (x1, y1) = t.ImageToView(new ImagePoint(4000, 3000));
        Assert.Equal(8, x0, 9); Assert.Equal(808, x1, 9);
        Assert.Equal(8, y0, 9); Assert.Equal(608, y1, 9);
    }

    [Fact]
    public void ActualPixels_AccountsForDpiScale()
    {
        var t = ViewTransform.ActualPixels(1.5, 0, 0, ViewTransform.Identity);
        Assert.Equal(1 / 1.5, t.Scale, 12); // 1 image px = 1 physical px at 144 DPI
    }
}

public class RectAndPolygonTests
{
    [Fact]
    public void RectOperations()
    {
        var a = new PixelRect(0, 0, 10, 10);
        var b = new PixelRect(5, 5, 10, 10);
        Assert.Equal(new PixelRect(5, 5, 5, 5), a.Intersect(b));
        Assert.Equal(new PixelRect(0, 0, 15, 15), a.Union(b));
        Assert.Equal(new PixelRect(0, 0, 4, 4), new PixelRect(-3, -3, 7, 7).ClampTo(100, 100));
        Assert.True(new PixelRect(200, 0, 5, 5).ClampTo(100, 100).IsEmpty);
        Assert.Equal(new PixelRect(2, 3, 5, 6), PixelRect.FromInclusivePixels(new PixelPoint(6, 8), new PixelPoint(2, 3)));
    }

    [Fact]
    public void ShoelaceAreaAndPerimeter()
    {
        var sq = new List<PixelPoint> { new(0, 0), new(10, 0), new(10, 10), new(0, 10) };
        Assert.Equal(100, PolygonMath.Area(sq));
        Assert.Equal(40, PolygonMath.Perimeter(sq));
        var comp = new PolygonComponent(sq, new[] { new[] { new PixelPoint(2, 2), new(4, 2), new(4, 4), new(2, 4) } });
        Assert.Equal(96, PolygonMath.Area(comp));
        Assert.Equal(new PixelRect(0, 0, 11, 11), PolygonMath.BoundingBox(sq));
    }

    [Fact]
    public void SelfIntersectionDetection()
    {
        var bowtie = new List<PixelPoint> { new(0, 0), new(10, 10), new(10, 0), new(0, 10) };
        var square = new List<PixelPoint> { new(0, 0), new(10, 0), new(10, 10), new(0, 10) };
        Assert.True(PolygonMath.IsSelfIntersecting(bowtie));
        Assert.False(PolygonMath.IsSelfIntersecting(square));
    }

    [Fact]
    public void DouglasPeucker_RemovesCollinearVertices()
    {
        var ring = new List<PixelPoint>();
        for (int x = 0; x < 20; x++) ring.Add(new(x, 0));
        for (int y = 0; y < 20; y++) ring.Add(new(20, y));
        for (int x = 20; x > 0; x--) ring.Add(new(x, 20));
        for (int y = 20; y > 0; y--) ring.Add(new(0, y));
        var s = PolygonMath.SimplifyClosed(ring, 0.5);
        Assert.Equal(4, s.Count);
        Assert.Equal(400, PolygonMath.Area(s));
    }
}

public class RasterMaskTests
{
    private static RasterMask Square(int x, int y, int n) => new(new PixelRect(x, y, n, n), Enumerable.Repeat((byte)1, n * n).ToArray());

    [Fact]
    public void BooleanOperations()
    {
        var a = Square(0, 0, 10); var b = Square(5, 5, 10);
        Assert.Equal(175, a.Union(b).PixelCount);
        Assert.Equal(25, a.Intersect(b).PixelCount);
        Assert.Equal(75, a.Subtract(b).PixelCount);
        Assert.Equal(25, a.OverlapCount(b));
        Assert.True(a.Intersect(Square(50, 50, 3)).IsEmpty);
    }

    [Fact]
    public void ReframeAndTrimPreserveContent()
    {
        var a = Square(3, 4, 5);
        var padded = a.Reframe(new PixelRect(0, 0, 20, 20));
        Assert.Equal(a.PixelCount, padded.PixelCount);
        Assert.True(a.Equals(padded));
        Assert.Equal(a.Bounds, padded.Trim().Bounds);
        Assert.Equal(new PixelRect(3, 4, 5, 5), padded.ContentBounds());
    }

    [Fact]
    public void ClipToImage_RemovesOutOfBoundsPixels()
    {
        var a = Square(-5, -5, 10).ClipToImage(100, 100);
        Assert.Equal(25, a.PixelCount);
        Assert.Equal(new PixelRect(0, 0, 5, 5), a.Bounds);
    }

    [Fact]
    public void CentroidAndPerimeter()
    {
        var a = Square(10, 20, 4);
        Assert.Equal((12.0, 22.0), a.Centroid());
        Assert.Equal(16, a.CrackPerimeter());
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void CocoRle_RoundTripsThroughCompressedString(int seed)
    {
        var rnd = new Random(seed);
        int w = 37, h = 23;
        var data = new byte[w * h];
        for (int i = 0; i < data.Length; i++) data[i] = rnd.NextDouble() < 0.3 ? (byte)1 : (byte)0;
        var mask = new RasterMask(new PixelRect(0, 0, w, h), data);
        var counts = CocoRle.EncodeLocal(mask);
        Assert.Equal(w * h, counts.Sum());
        var s = CocoRle.ToCompressedString(counts);
        Assert.Equal(counts, CocoRle.FromCompressedString(s));
        Assert.True(mask.Equals(CocoRle.DecodeLocal(CocoRle.FromCompressedString(s), mask.Bounds)));
    }

    [Theory]
    [InlineData(0, 0, 10, 8)]   // touches top-left
    [InlineData(5, 3, 10, 8)]   // interior
    [InlineData(30, 22, 10, 8)] // touches bottom-right
    [InlineData(0, 0, 40, 30)]  // full frame
    [InlineData(-4, 25, 12, 9)] // partly outside (clipped)
    public void CocoRle_BoundedEncoder_MatchesGenericEncoder(int x, int y, int w, int h)
    {
        var rnd = new Random(x * 31 + y);
        var data = new byte[w * h];
        for (int i = 0; i < data.Length; i++) data[i] = rnd.NextDouble() < 0.5 ? (byte)1 : (byte)0;
        var mask = new RasterMask(new PixelRect(x, y, w, h), data);
        Assert.Equal(CocoRle.Encode(mask.Get, 40, 30), CocoRle.Encode(mask, 40, 30));
    }

    [Fact]
    public void CocoRle_UsesColumnMajorOrderStartingWithZeros()
    {
        // 2×2 mask with only (x=1, y=0) set → column-major sequence [0,0,1,0] → counts [2,1,1].
        var m = new RasterMask(new PixelRect(0, 0, 2, 2), new byte[] { 0, 1, 0, 0 });
        Assert.Equal(new long[] { 2, 1, 1 }, CocoRle.EncodeLocal(m));
    }

    [Fact]
    public void CocoRle_KnownPycocotoolsString()
    {
        // pycocotools: encode(np.asfortranarray([[0,1],[0,0]], dtype=uint8))['counts'] == b'211'
        Assert.Equal("211", CocoRle.ToCompressedString(new long[] { 2, 1, 1 }));
    }
}

public class TaxonomyTests
{
    [Fact]
    public void DefaultTaxonomy_IsValid_AndMatchesResearchDatasetIds()
    {
        var t = Taxonomy.CreateDefault();
        Assert.Empty(TaxonomyValidator.Validate(t));
        Assert.Equal("Unlabeled", t.Get(0).Name);
        Assert.Equal("Healthy Copper", t.Get(1).Name);
        Assert.Equal("Cu2O", t.Get(2).Name);
        Assert.Equal("CuO", t.Get(3).Name);
        Assert.Equal("CuCl", t.Get(4).Name);
        Assert.Equal("CuCl2", t.Get(5).Name);
        Assert.Equal("Malachite", t.Get(6).Name);
        Assert.Contains("sulfate", t.Get(7).Description);
        Assert.Equal("Ignore", t.Get(255).Name);
        Assert.Equal("I", t.Get(255).Shortcut);
    }

    [Fact]
    public void Validator_DetectsDuplicatesReservedConflictsAndBadColours()
    {
        var t = Taxonomy.CreateDefault();
        t.Classes.Add(new CorrosionClass { Id = 2, Name = "dup", ColorHex = "#112233" });
        t.Classes.Add(new CorrosionClass { Id = 255, Name = "fake ignore", ColorHex = "#112233" });
        t.Classes.Add(new CorrosionClass { Id = 20, Name = "bad", ColorHex = "red", Shortcut = "2" });
        var errors = TaxonomyValidator.Validate(t);
        Assert.Contains(errors, e => e.Contains("ID 2 is used by more than one"));
        Assert.Contains(errors, e => e.Contains("reserved"));
        Assert.Contains(errors, e => e.Contains("not #RRGGBB"));
        Assert.Contains(errors, e => e.Contains("Shortcut '2'"));
    }

    [Fact]
    public void ValidateEdit_RejectsRemovingClassesInUse()
    {
        var before = Taxonomy.CreateDefault();
        var after = before.Clone();
        after.Classes.RemoveAll(c => c.Id == 3);
        Assert.Contains(TaxonomyValidator.ValidateEdit(before, after, new[] { 3 }), e => e.Contains("missing"));
        Assert.Empty(TaxonomyValidator.ValidateEdit(before, after, new[] { 2 }));
    }

    [Fact]
    public void Annotation_DeepCloneAndRestoreFromAreIndependent()
    {
        var a = new Annotation { ClassId = 2, Notes = "x", Evidence = { new EvidenceReference { Source = "s" } } };
        var c = a.DeepClone();
        c.Notes = "y"; c.Evidence[0].Source = "t";
        Assert.Equal("x", a.Notes);
        Assert.Equal("s", a.Evidence[0].Source);
        var id = a.Id;
        a.RestoreFrom(c);
        Assert.Equal("y", a.Notes);
        Assert.Equal(id, a.Id);
    }
}
