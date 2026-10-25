using CuAnnotator.Application.Annotations;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

// Generates samples/SampleProject: synthetic copper-corrosion images annotated through the real
// segmentation pipeline (no hand-drawn or hard-coded annotation geometry).
// Usage: dotnet run --project samples/CuAnnotator.SampleGenerator -- [outputFolder]

var output = Path.GetFullPath(args.Length > 0 ? args[0] : Path.Combine("samples", "SampleProject"));
if (Directory.Exists(output)) Directory.Delete(output, true);
var work = Path.Combine(Path.GetTempPath(), "cuann-sample-src");
Directory.CreateDirectory(work);

var serializer = new NativeJsonSerializer();
var store = new ProjectStore(serializer);
var projects = new ProjectService(store);
var project = projects.Create(new NewProjectRequest
{
    Name = "Synthetic copper corrosion example",
    Folder = output,
    Researcher = "CuAnnotator sample generator",
    Description = "SYNTHETIC images and annotations for demonstrating the workflow. Colours are illustrative; " +
                  "no chemical identification is implied. Evidence records are explicitly marked as fictitious examples.",
});

var specimens = new[] { ("specimen_A_01.png", "A", 1), ("specimen_B_01.png", "B", 2) };
foreach (var (file, specimen, seed) in specimens)
{
    var path = Path.Combine(work, file);
    using (var img = SyntheticCopper.Render(1024, 768, seed)) Cv2.ImWrite(path, img);
    await projects.ImportImagesAsync(project, output, new[] { path }, specimen, "Synthetic (generated)");
}

var audit = store.OpenAuditLog(output);
var engine = new SegmentationEngine();
foreach (var item in project.Images)
{
    using var loaded = ProjectService.LoadImage(output, item, out _);
    var s = new AnnotationSession(project, item, loaded.Bgr, engine, audit, "sample-generator");
    var layout = SyntheticCopper.Layout(item.SpecimenId == "A" ? 1 : 2);

    // Cu2O: loose rectangle → GrabCut → background correction on the neighbouring patch → refine.
    s.BeginRoi(layout.OxideRoi, new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.GrabCut });
    await s.RunSegmentationAsync();
    s.AddGuidance(GuidanceKind.Foreground, Stroke.Point(layout.OxideSeed, 4));
    s.AddGuidance(GuidanceKind.Background, Stroke.Point(layout.IslandCenter, 10)); // exposed copper island → hole
    await s.RunSegmentationAsync();
    var cu2o = s.AcceptCandidate(2, VerificationStatus.VisuallyEstimated, "Dark red layer with an exposed copper island (hole).").Annotation;
    ClassSuggester.ComputeDescriptor(loaded.Bgr, cu2o);

    // CuO: colour-based selection → several disconnected patches in one annotation.
    var cuo = await engine.ColorSelectAsync(loaded.Bgr, layout.CuORegion, new[] { Stroke.Point(layout.CuOSeed, 2) },
        new SegmentationParameters { ColorThreshold = 10, MinComponentArea = 20 }, default);
    s.ProposeCandidate(cuo, "Colour-Based Selection");
    var cuoAnn = s.AcceptCandidate(3, VerificationStatus.Unverified, "Dark patches selected by CIEDE2000 similarity.").Annotation;
    ClassSuggester.ComputeDescriptor(loaded.Bgr, cuoAnn);

    // Malachite: hybrid pipeline (GrabCut + CIELAB + edge refinement).
    s.BeginRoi(layout.MalachiteRoi, new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.Hybrid });
    await s.RunSegmentationAsync();
    var mal = s.AcceptCandidate(6, VerificationStatus.VisuallyEstimated).Annotation;
    s.SetVerification(mal.Id, VerificationStatus.ExpertReviewed, "example reviewer",
        new EvidenceReference { Type = EvidenceType.ExpertObservation, Source = "FICTITIOUS EXAMPLE — not a real measurement", Notes = "Demonstrates evidence recording." });
    ClassSuggester.ComputeDescriptor(loaded.Bgr, mal);

    // Basic copper chlorides: magic wand on small powdery spots (each wand click is one annotation).
    foreach (var p in layout.ChlorideSeeds)
    {
        var w = await engine.MagicWandAsync(loaded.Bgr, p, new SegmentationParameters { ColorThreshold = 9, MinComponentArea = 3 }, 120, default);
        s.ProposeCandidate(w, "Magic Wand Selection");
        ClassSuggester.ComputeDescriptor(loaded.Bgr, s.AcceptCandidate(8, VerificationStatus.Unverified).Annotation);
    }

    // Healthy copper: CIELAB region growing from a seed inside a clean area.
    s.BeginRoi(layout.HealthyRoi, new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.LabColor, ColorThreshold = 8 });
    s.AddGuidance(GuidanceKind.Foreground, Stroke.Point(layout.HealthySeed, 6));
    await s.RunSegmentationAsync();
    ClassSuggester.ComputeDescriptor(loaded.Bgr, s.AcceptCandidate(1, VerificationStatus.VisuallyEstimated).Annotation);

    item.WorkflowState = ImageWorkflowState.Annotated;
    Console.WriteLine($"{item.FileName}: {item.Annotations.Count} annotations; " +
                      string.Join(", ", item.Annotations.Select(a => $"{project.Taxonomy.Get(a.ClassId).Name}={a.PixelArea}px/{a.Polygons.Count}c/{a.Polygons.Sum(p => p.Holes.Count)}h")));
}

await projects.SaveAsync(project, output);
var report = await new DatasetExporter(serializer).ExportAsync(project, output, new DatasetExportOptions
{
    OutputFolder = Path.Combine(output, "exports", "dataset_example"), IncludeYolo = true, IncludeAllCorrosionBinary = true,
    TrainFraction = 0.5, ValidationFraction = 0.0,
});
Console.WriteLine($"Exported {report.FilesWritten.Count} files; {report.Warnings.Count} warning(s).");
foreach (var w in report.Warnings) Console.WriteLine("  warning: " + w);
Directory.Delete(work, true);
Console.WriteLine($"Sample project written to {output}");

/// <summary>Procedural copper surface with corrosion-like regions (deterministic per seed).</summary>
internal static class SyntheticCopper
{
    public sealed record SceneLayout(PixelRect OxideRoi, PixelPoint OxideSeed, PixelRect CuORegion, PixelPoint CuOSeed,
        PixelRect MalachiteRoi, PixelPoint[] ChlorideSeeds, PixelRect HealthyRoi, PixelPoint HealthySeed)
    {
        public PixelPoint IslandCenter => new(OxideSeed.X + 50, OxideSeed.Y + 45);
    }

    public static SceneLayout Layout(int seed) => seed == 1
        ? new(new PixelRect(150, 120, 330, 280), new PixelPoint(300, 200), new PixelRect(560, 60, 420, 260), new PixelPoint(640, 120),
              new PixelRect(560, 420, 330, 280), new[] { new PixelPoint(160, 560), new PixelPoint(260, 640), new PixelPoint(380, 590) },
              new PixelRect(40, 470, 90, 240), new PixelPoint(85, 590))
        : new(new PixelRect(500, 380, 330, 280), new PixelPoint(650, 460), new PixelRect(60, 60, 420, 260), new PixelPoint(140, 120),
              new PixelRect(120, 420, 330, 280), new[] { new PixelPoint(620, 120), new PixelPoint(720, 200), new PixelPoint(840, 150) },
              new PixelRect(880, 420, 110, 300), new PixelPoint(935, 570));

    public static Mat Render(int w, int h, int seed)
    {
        var rnd = new Random(seed * 7919);
        var img = new Mat(h, w, MatType.CV_8UC3);
        // Copper base with illumination gradient and brushed texture.
        var px = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double light = 0.85 + 0.25 * (x + y) / (double)(w + h) + 0.04 * Math.Sin(y * 0.9 + Math.Sin(x * 0.01) * 3);
                int i = (y * w + x) * 3;
                px[i] = Clamp(75 * light); px[i + 1] = Clamp(125 * light); px[i + 2] = Clamp(195 * light);
            }
        System.Runtime.InteropServices.Marshal.Copy(px, 0, img.Data, px.Length);
        var L = Layout(seed);

        // Cu2O-like irregular layer with a copper island (hole).
        var oxide = Blob(rnd, L.OxideRoi.Inflate(-45), 14);
        Cv2.FillPoly(img, new[] { oxide }, new Scalar(35, 38, 140), LineTypes.Link8);
        var hole = new Point(L.IslandCenter.X, L.IslandCenter.Y);
        Cv2.Ellipse(img, hole, new Size(26, 18), 20, 0, 360, new Scalar(80, 130, 200), -1, LineTypes.Link8);

        // CuO-like dark disconnected patches.
        for (int k = 0; k < 4; k++)
        {
            var r = L.CuORegion;
            var c = k == 0 ? new Point(L.CuOSeed.X, L.CuOSeed.Y) : new Point(r.X + 60 + rnd.Next(r.Width - 120), r.Y + 50 + rnd.Next(r.Height - 100));
            Cv2.FillPoly(img, new[] { Blob(rnd, new PixelRect(c.X - 40, c.Y - 30, 80, 60), 10) }, new Scalar(28, 30, 34), LineTypes.Link8);
        }

        // Malachite-like green region with speckle texture.
        var mal = Blob(rnd, L.MalachiteRoi.Inflate(-40), 16);
        Cv2.FillPoly(img, new[] { mal }, new Scalar(95, 150, 55), LineTypes.Link8);
        for (int k = 0; k < 400; k++)
        {
            var r = L.MalachiteRoi.Inflate(-60);
            var p = new Point(r.X + rnd.Next(r.Width), r.Y + rnd.Next(r.Height));
            if (Cv2.PointPolygonTest(mal, p, false) > 0) Cv2.Circle(img, p, 1, new Scalar(80, 135, 45), -1);
        }

        // Pale green chloride spots.
        foreach (var p in L.ChlorideSeeds)
            Cv2.FillPoly(img, new[] { Blob(rnd, new PixelRect(p.X - 14, p.Y - 12, 28, 24), 9) }, new Scalar(150, 215, 140), LineTypes.Link8);

        // Sensor noise.
        Cv2.SetTheRNG((ulong)seed);
        using var noise = new Mat(img.Size(), MatType.CV_16SC3);
        Cv2.Randn(noise, Scalar.All(0), Scalar.All(3));
        using var tmp = new Mat();
        img.ConvertTo(tmp, MatType.CV_16SC3); Cv2.Add(tmp, noise, tmp); tmp.ConvertTo(img, MatType.CV_8UC3);
        Cv2.GaussianBlur(img, img, new Size(3, 3), 0.6);
        return img;
    }

    private static Point[] Blob(Random rnd, PixelRect r, int n)
    {
        double cx = r.X + r.Width / 2.0, cy = r.Y + r.Height / 2.0;
        return Enumerable.Range(0, n).Select(i =>
        {
            double a = 2 * Math.PI * i / n, k = 0.65 + 0.35 * rnd.NextDouble();
            return new Point((int)(cx + Math.Cos(a) * r.Width / 2 * k), (int)(cy + Math.Sin(a) * r.Height / 2 * k));
        }).ToArray();
    }

    private static byte Clamp(double v) => (byte)Math.Clamp(Math.Round(v), 0, 255);
}
