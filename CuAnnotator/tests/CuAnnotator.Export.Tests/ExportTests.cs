using System.Text.Json;
using CuAnnotator.Application.Evaluation;
using CuAnnotator.Application.Export;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Vision.ImageIO;

namespace CuAnnotator.Export.Tests;

public sealed class ExportTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "cuann-export-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { try { Directory.Delete(_root, true); } catch (IOException) { } }

    private static Annotation Ann(ImageItem img, int cls, RasterMask m, int z = 0)
    {
        var a = new Annotation { ImageId = img.Id, ClassId = cls, ZOrder = z, CreatedAt = DateTimeOffset.UnixEpoch.AddSeconds(z) };
        Application.Annotations.AnnotationGeometryService.SetMask(a, m);
        img.Annotations.Add(a);
        return a;
    }

    private static RasterMask Rect(int x, int y, int w, int h) => new(new PixelRect(x, y, w, h), Enumerable.Repeat((byte)1, w * h).ToArray());

    private static RasterMask Ring(int cx, int cy, int r1, int r2)
    {
        var b = new PixelRect(cx - r2, cy - r2, 2 * r2 + 1, 2 * r2 + 1);
        var d = new byte[b.Width * b.Height];
        for (int y = 0; y < b.Height; y++)
            for (int x = 0; x < b.Width; x++)
            {
                double dd = Math.Sqrt(Math.Pow(x + b.X - cx, 2) + Math.Pow(y + b.Y - cy, 2));
                d[y * b.Width + x] = dd <= r2 && dd > r1 ? (byte)1 : (byte)0;
            }
        return new RasterMask(b, d);
    }

    [Fact]
    public void SemanticLabelMap_UsesClassIds_ZOrder_AndIgnore()
    {
        var img = new ImageItem { Width = 50, Height = 40 };
        Ann(img, 2, Rect(0, 0, 20, 20), z: 0);
        Ann(img, 6, Rect(10, 10, 20, 20), z: 1);
        Ann(img, 255, Rect(40, 30, 5, 5), z: 2);
        var map = LabelMapComposer.Compose(img);
        Assert.Equal(2, map.Labels[5 * 50 + 5]);
        Assert.Equal(6, map.Labels[15 * 50 + 15]); // topmost wins
        Assert.Equal(255, map.Labels[32 * 50 + 42]);
        Assert.Equal(0, map.Labels[39 * 50 + 0]);
        Assert.Equal(100, map.OverlapPixels);
    }

    [Fact]
    public void LabelPng_IsLosslessSingleChannel()
    {
        var labels = new byte[64 * 48];
        var rnd = new Random(1);
        for (int i = 0; i < labels.Length; i++) labels[i] = (byte)new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 255 }[rnd.Next(10)];
        var p = Path.Combine(_root, "m.png");
        MaskIO.WriteLabelPng(p, labels, 64, 48);
        var (back, w, h) = MaskIO.ReadLabelPng(p);
        Assert.Equal((64, 48), (w, h));
        Assert.Equal(labels, back);
    }

    [Fact]
    public void Coco_UsesRleForHoles_PolygonsOtherwise_AndRleDecodesExactly()
    {
        var project = new AnnotationProject { Name = "p" };
        var img = new ImageItem { Width = 120, Height = 100, FileName = "a.png" };
        project.Images.Add(img);
        var ring = Ann(img, 2, Ring(40, 40, 10, 25));
        var rect = Ann(img, 3, Rect(80, 10, 30, 20), 1);
        var report = new ExportReport();
        var ds = CocoExporter.Build(project, new[] { (img, "a.png") }, CocoSegmentationMode.Auto, report);
        var json = JsonDocument.Parse(JsonSerializer.Serialize(ds)).RootElement;
        var anns = json.GetProperty("annotations");
        Assert.Equal(2, anns.GetArrayLength());

        var rle = anns[0].GetProperty("segmentation");
        Assert.Equal(JsonValueKind.Object, rle.ValueKind);
        Assert.Equal(new[] { 100, 120 }, rle.GetProperty("size").EnumerateArray().Select(e => e.GetInt32()));
        var decoded = CocoRle.Decode(CocoRle.FromCompressedString(rle.GetProperty("counts").GetString()!), 120, 100);
        Assert.True(new RasterMask(new PixelRect(0, 0, 120, 100), decoded).Equals(ring.Mask));
        Assert.Equal(ring.PixelArea, anns[0].GetProperty("area").GetDouble());

        var poly = anns[1].GetProperty("segmentation");
        Assert.Equal(JsonValueKind.Array, poly.ValueKind);
        Assert.Equal(new double[] { 80, 10, 30, 20 }, anns[1].GetProperty("bbox").EnumerateArray().Select(e => e.GetDouble()));
        Assert.Equal(600, anns[1].GetProperty("area").GetDouble());
        Assert.Contains(report.Messages, m => m.Contains("holes"));
        Assert.DoesNotContain(json.GetProperty("categories").EnumerateArray(), c => c.GetProperty("id").GetInt32() is 0 or 255);
    }

    [Fact]
    public void Yolo_NormalisesCoordinates_AndWarnsAboutHolesAndMultipart()
    {
        var tax = Taxonomy.CreateDefault();
        var img = new ImageItem { Width = 200, Height = 100, FileName = "y.png" };
        Ann(img, 2, Rect(10, 10, 20, 10));
        Ann(img, 3, Ring(100, 50, 5, 20), 1);
        var multi = Rect(150, 10, 5, 5).Union(Rect(170, 10, 5, 5));
        Ann(img, 6, multi, 2);
        var report = new ExportReport();
        var map = YoloExporter.ClassIndexMap(tax);
        var text = YoloExporter.LabelFile(img, map, YoloHolePolicy.Skip, report);
        var lines = text.Trim().Split('\n');
        Assert.Equal(3, lines.Length); // rect + 2 parts of the multipart; ring skipped
        var first = lines[0].Split(' ');
        Assert.Equal(map[2].ToString(), first[0]);
        Assert.All(first.Skip(1).Select(double.Parse), v => Assert.InRange(v, 0, 1));
        Assert.Equal((10 + 0.5) / 200, double.Parse(first[1]), 6);
        Assert.Contains(report.Warnings, w => w.Contains("holes") && w.Contains("skipped"));
        Assert.Contains(report.Warnings, w => w.Contains("disconnected parts"));
    }

    [Fact]
    public void Splitter_IsReproducible_AndNeverLeaksSpecimens()
    {
        var images = Enumerable.Range(0, 60).Select(i => new ImageItem { FileName = $"{i}.png", SpecimenId = $"S{i / 4}" }).ToList();
        var a = DatasetSplitter.Split(images, 0.7, 0.15, 123);
        var b = DatasetSplitter.Split(images, 0.7, 0.15, 123);
        Assert.Equal(a.OrderBy(k => k.Key), b.OrderBy(k => k.Key));
        Assert.Empty(DatasetSplitter.LeakingSpecimens(images, a));
        var trainCount = a.Values.Count(v => v == DatasetSplit.Train);
        Assert.InRange(trainCount, 36, 48);
        Assert.Contains(a.Values, v => v == DatasetSplit.Test);
        var ungrouped = DatasetSplitter.Split(images, 0.7, 0.15, 123, groupBySpecimen: false);
        Assert.NotEmpty(DatasetSplitter.LeakingSpecimens(images, ungrouped));
    }

    [Fact]
    public async Task DatasetExport_WritesLayout_AndValidatesMasks()
    {
        var src = Path.Combine(_root, "proj");
        Directory.CreateDirectory(Path.Combine(src, "images"));
        using (var m = new OpenCvSharp.Mat(60, 80, OpenCvSharp.MatType.CV_8UC3, new OpenCvSharp.Scalar(10, 100, 200)))
            OpenCvSharp.Cv2.ImWrite(Path.Combine(src, "images", "s 1.png"), m);
        var project = new AnnotationProject { Name = "p" };
        var img = new ImageItem { Width = 80, Height = 60, FileName = "s 1.png", RelativePath = "images/s 1.png", SpecimenId = "A" };
        project.Images.Add(img);
        Ann(img, 4, Rect(5, 5, 10, 10));
        Ann(img, 1, Rect(30, 5, 10, 10), 1);
        var outDir = Path.Combine(_root, "out");
        var report = await new DatasetExporter(new NativeJsonSerializer()).ExportAsync(project, src, new DatasetExportOptions
        {
            OutputFolder = outDir, IncludeYolo = true, BinaryMaskClasses = new[] { 4 }, IncludeAllCorrosionBinary = true, CreateSplits = false,
        });
        Assert.Empty(report.Warnings);
        foreach (var rel in new[] { "images/s_1.png", "masks/s_1.png", "colorized_masks/s_1.png", "annotations/s_1.json", "coco/annotations.json",
                     "yolo/data.yaml", "yolo/labels/train/s_1.txt", "metadata/classes.json", "metadata/dataset.json", "metadata/verification.json",
                     "README.md", "binary_masks/class_4/s_1.png", "binary_masks/all_corrosion/s_1.png" })
            Assert.True(File.Exists(Path.Combine(outDir, rel)), rel);
        var (bin, _, _) = MaskIO.ReadLabelPng(Path.Combine(outDir, "binary_masks/all_corrosion/s_1.png"));
        Assert.Equal(100, bin.Count(v => v == 255)); // CuCl only; healthy copper excluded
    }

    [Fact]
    public void Metrics_IoUDicePrecisionRecall()
    {
        var p = new byte[] { 1, 1, 0, 0, 1, 0 };
        var r = new byte[] { 1, 0, 0, 1, 1, 0 };
        var m = SegmentationMetrics.Binary(p, r);
        Assert.Equal((2L, 1L, 1L, 2L), (m.TP, m.FP, m.FN, m.TN));
        Assert.Equal(0.5, m.IoU, 9);
        Assert.Equal(2.0 / 3, m.Dice, 9);
        Assert.Equal(2.0 / 3, m.Precision, 9);
        Assert.Equal(2.0 / 3, m.Recall, 9);
    }

    [Fact]
    public void BoundaryF1_IsOneForIdenticalAndToleratesSmallShifts()
    {
        int w = 40, h = 40;
        byte[] Sq(int off) { var a = new byte[w * h]; for (int y = 10; y < 30; y++) for (int x = 10 + off; x < 30 + off; x++) a[y * w + x] = 1; return a; }
        Assert.Equal(1, SegmentationMetrics.BoundaryF1(Sq(0), Sq(0), w, h, 0).F1, 9);
        Assert.Equal(1, SegmentationMetrics.BoundaryF1(Sq(1), Sq(0), w, h, 1.0).F1, 9);
        Assert.True(SegmentationMetrics.BoundaryF1(Sq(5), Sq(0), w, h, 1.0).F1 < 0.6);
    }

    [Fact]
    public void CohenKappa_PerfectAndChanceAgreement()
    {
        var a = new byte[] { 1, 2, 3, 1, 2, 3 };
        Assert.Equal(1, SegmentationMetrics.Agreement(a, a).CohenKappa, 9);
        var x = new byte[] { 1, 1, 2, 2 };
        var y = new byte[] { 1, 2, 1, 2 };
        Assert.Equal(0, SegmentationMetrics.Agreement(x, y).CohenKappa, 9);
    }

    [Fact]
    public void QualityAnalyzer_ReportsCoverageAndIssues()
    {
        var project = new AnnotationProject();
        var img = new ImageItem { Width = 10, Height = 10, FileName = "q.png" };
        project.Images.Add(img);
        Ann(img, 2, Rect(0, 0, 5, 10));
        Ann(img, 255, Rect(5, 0, 2, 10), 1);
        var bad = Ann(img, 42, Rect(8, 0, 1, 1), 2);
        var r = QualityAnalyzer.Analyze(project, project.Images);
        Assert.Equal(51, r.LabeledPixels); // 50 Cu2O + 1 px of undefined class 42
        Assert.Equal(20, r.IgnoredPixels);
        Assert.Equal(51.0, r.CoveragePercent, 6);
        Assert.Contains(r.Issues, i => i.AnnotationId == bad.Id && i.Severity == Application.Evaluation.IssueSeverity.Error);
        Assert.Contains(r.Issues, i => i.Message.Contains("is Unverified; class expects"));
    }

    [Fact]
    public void NativeJson_StoresMasksAsCompressedRleWithIntegrityCheck()
    {
        var ser = new NativeJsonSerializer();
        var project = new AnnotationProject { Name = "n" };
        var img = new ImageItem { Width = 60, Height = 60, FileName = "n.png" };
        project.Images.Add(img);
        var a = Ann(img, 2, Ring(30, 30, 8, 20));
        a.VerificationStatus = VerificationStatus.ExpertReviewed;
        var json = ser.Serialize(project);
        Assert.Contains("\"rle\"", json);
        Assert.Contains("\"ExpertReviewed\"", json);
        var back = ser.Deserialize(json);
        Assert.True(back.Images[0].Annotations[0].Mask.Equals(a.Mask));
        Assert.Equal(a.Polygons[0].Holes.Count, back.Images[0].Annotations[0].Polygons[0].Holes.Count);
        var tampered = json.Replace($"\"pixels\": {a.PixelArea}", $"\"pixels\": {a.PixelArea + 1}");
        Assert.Throws<InvalidDataException>(() => ser.Deserialize(tampered));
    }
}
