using System.Text.Json;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Contracts;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Infrastructure.SQLite;
using CuAnnotator.Vision.ImageIO;
using CuAnnotator.Vision.Interop;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Integration.Tests;

/// <summary>
/// Section 27 — "Annotating a Cu₂O corrosion region", executed headlessly through the same application
/// services the WPF front-end calls (the UI only translates mouse/keyboard input into these calls).
/// </summary>
public sealed class AcceptanceScenarioTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "cuann-accept-" + Guid.NewGuid().ToString("N"));

    public void Dispose() { try { Directory.Delete(_root, true); } catch (IOException) { } }

    /// <summary>High-resolution synthetic copper image: an irregular oxide region next to a second, similar oxide blob.</summary>
    private static (string Path, RasterMask Target) MakeImage(string folder)
    {
        int w = 3000, h = 2000;
        using var img = new Mat(h, w, MatType.CV_8UC3, new Scalar(70, 120, 190));
        var target = new[] { new Point(1200, 800), new Point(1420, 760), new Point(1530, 900), new Point(1450, 950), new Point(1520, 1120), new Point(1260, 1150), new Point(1310, 980) };
        Cv2.FillPoly(img, new[] { target }, new Scalar(40, 40, 150), LineTypes.Link8);
        // A similar oxide blob touching the region the user will roughly select (unwanted).
        Cv2.Circle(img, new Point(1610, 1000), 60, new Scalar(42, 42, 148), -1, LineTypes.Link8);
        Cv2.SetTheRNG(99);
        using var noise = new Mat(img.Size(), MatType.CV_16SC3);
        Cv2.Randn(noise, Scalar.All(0), Scalar.All(5));
        using var tmp = new Mat();
        img.ConvertTo(tmp, MatType.CV_16SC3); Cv2.Add(tmp, noise, tmp); tmp.ConvertTo(img, MatType.CV_8UC3);
        Directory.CreateDirectory(folder);
        var path = System.IO.Path.Combine(folder, "specimen_001.png");
        Cv2.ImWrite(path, img);
        using var m = new Mat(h, w, MatType.CV_8UC1, Scalar.All(0));
        Cv2.FillPoly(m, new[] { target }, Scalar.All(1), LineTypes.Link8);
        return (path, MatInterop.MatToMask(m, new PixelRect(0, 0, w, h)));
    }

    private static double IoU(RasterMask a, RasterMask b)
    {
        long i = a.OverlapCount(b);
        return (double)i / (a.PixelCount + b.PixelCount - i);
    }

    [Fact]
    public async Task MandatoryEndToEndScenario_Cu2O()
    {
        var serializer = new NativeJsonSerializer();
        var store = new ProjectStore(serializer);
        var projects = new ProjectService(store);
        var projectFolder = Path.Combine(_root, "Project");

        // 1–3. Launch, create project, import a high-resolution image.
        var project = projects.Create(new NewProjectRequest { Name = "Cu corrosion study", Folder = projectFolder, Researcher = "Test Researcher" });
        var (imagePath, truth) = MakeImage(Path.Combine(_root, "incoming"));
        var import = await projects.ImportImagesAsync(project, projectFolder, new[] { imagePath }, specimenId: "S-001");
        var item = Assert.Single(import.Imported);
        Assert.Equal((3000, 2000), (item.Width, item.Height));
        Assert.Equal(64, item.Sha256.Length);
        Assert.Equal("images/specimen_001.png", item.RelativePath);

        // 4. Display (decode) the image.
        using var loaded = ProjectService.LoadImage(projectFolder, item, out var warn);
        Assert.Null(warn);
        var audit = store.OpenAuditLog(projectFolder);
        var session = new AnnotationSession(project, item, loaded.Bgr, new SegmentationEngine(), audit, "tester");

        // 5–8. Loose rectangle around the oxidised region (it also clips the neighbouring blob), run GrabCut.
        var roi = PixelRect.FromCorners(1150, 720, 1640, 1200);
        session.BeginRoi(roi, new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.GrabCut });
        var first = await session.RunSegmentationAsync();
        Assert.NotNull(first);
        Assert.True(first!.Mask.PixelCount < 0.8 * roi.Area, "GrabCut must produce an irregular region, not the rectangle");
        Assert.True(first.Components.Sum(c => c.Outer.Count) > 50);
        double iouFirst = IoU(first.Mask, truth);

        // 9–11. Foreground correction on target pixels; background correction on the unwanted blob; rerun.
        session.AddGuidance(GuidanceKind.Foreground, new Stroke(new[] { new PixelPoint(1300, 900), new PixelPoint(1420, 1050) }, 6));
        session.AddGuidance(GuidanceKind.Background, new Stroke(new[] { new PixelPoint(1585, 960), new PixelPoint(1625, 1040) }, 12));
        var refined = await session.RunSegmentationAsync();
        Assert.NotNull(refined);
        double iouRefined = IoU(refined!.Mask, truth);
        Assert.True(iouRefined >= iouFirst - 1e-9, $"refinement must not degrade: {iouFirst:0.000} → {iouRefined:0.000}");
        Assert.True(iouRefined > 0.95, $"refined IoU {iouRefined:0.000}");
        Assert.False(refined.Mask.Get(1610, 1000), "background-corrected blob must be excluded");

        // 12–15. Accept, assign Cu2O (ID 2), Visually Estimated, notes.
        var accepted = session.AcceptCandidate(2, VerificationStatus.VisuallyEstimated, "Dark red cuprite-like layer; optical only.");
        var ann = accepted.Annotation;
        Assert.Equal(2, ann.ClassId);
        Assert.Equal(VerificationStatus.VisuallyEstimated, ann.VerificationStatus);
        Assert.Equal("GrabCut", ann.SegmentationAlgorithm);
        Assert.Equal("5", ann.AlgorithmParameters["GrabCutIterations"]);
        var acceptedMask = ann.Mask;

        // Undo/redo of acceptance keeps the model consistent.
        session.Undo();
        Assert.Empty(item.Annotations);
        Assert.NotNull(session.Candidate);
        session.Redo();
        Assert.Same(ann, Assert.Single(item.Annotations));

        // 16–17. Save; close.
        await projects.SaveAsync(project, projectFolder);
        var annotationId = ann.Id;
        session = null!;
        project = null!;

        // 18–19. Reopen; locate the annotation.
        var reopened = await new ProjectService(new ProjectStore(serializer)).OpenAsync(projectFolder);
        var item2 = reopened.Images.Single();
        var ann2 = item2.Annotations.Single(a => a.Id == annotationId);
        Assert.True(ann2.Mask.Equals(acceptedMask));
        Assert.Equal("Dark red cuprite-like layer; optical only.", ann2.Notes);

        // 20–21. Edit geometry (drag a vertex outward), creating revision 2; save.
        using var loaded2 = ProjectService.LoadImage(projectFolder, item2, out _);
        var session2 = new AnnotationSession(reopened, item2, loaded2.Bgr, new SegmentationEngine(), new ProjectStore(serializer).OpenAuditLog(projectFolder), "tester");
        var polys = ann2.Polygons.Select(p => p.Clone()).ToList();
        int top = polys[0].Outer.Select((p, i) => (p, i)).OrderBy(t => t.p.Y).First().i;
        polys[0].Outer[top] = new PixelPoint(polys[0].Outer[top].X, polys[0].Outer[top].Y - 30);
        session2.EditPolygons(annotationId, polys);
        Assert.Equal(2, ann2.Revision);
        Assert.Single(ann2.RevisionHistory);
        Assert.True(ann2.PixelArea > acceptedMask.PixelCount);
        var editedMask = ann2.Mask;
        await new ProjectService(new ProjectStore(serializer)).SaveAsync(reopened, projectFolder);

        // 22–24. Export semantic mask, native JSON and COCO.
        var outDir = Path.Combine(_root, "Dataset");
        var report = await new DatasetExporter(serializer).ExportAsync(reopened, projectFolder, new DatasetExportOptions
        {
            OutputFolder = outDir, IncludeCoco = true, IncludeNativeJson = true, IncludeSemanticMasks = true, CreateSplits = false,
        });
        Assert.DoesNotContain(report.Warnings, w => w.Contains("differ") || w.Contains("does not match"));

        // 25–27. Reopen the exported mask: dimensions match; every accepted Cu2O pixel has value 2, all others 0.
        var (labels, mw, mh) = MaskIO.ReadLabelPng(Path.Combine(outDir, "masks", "specimen_001.png"));
        Assert.Equal((3000, 2000), (mw, mh));
        long twos = 0;
        for (int y = 0; y < mh; y++)
            for (int x = 0; x < mw; x++)
            {
                byte v = labels[y * mw + x];
                bool inMask = editedMask.Get(x, y);
                Assert.True(inMask ? v == 2 : v == 0, $"pixel ({x},{y}) = {v}, inMask = {inMask}");
                if (v == 2) twos++;
            }
        Assert.Equal(editedMask.PixelCount, twos);

        // COCO validity: RLE or polygons decode to the same area; bbox and category correct.
        var coco = JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "coco", "annotations.json"))).RootElement;
        var ca = coco.GetProperty("annotations")[0];
        Assert.Equal(2, ca.GetProperty("category_id").GetInt32());
        Assert.Equal(editedMask.PixelCount, ca.GetProperty("area").GetDouble());
        var bb = editedMask.ContentBounds();
        Assert.Equal(new double[] { bb.X, bb.Y, bb.Width, bb.Height }, ca.GetProperty("bbox").EnumerateArray().Select(e => e.GetDouble()).ToArray());
        Assert.Contains(coco.GetProperty("categories").EnumerateArray(), c => c.GetProperty("id").GetInt32() == 2 && c.GetProperty("name").GetString() == "Cu2O");

        // 28. Native JSON retains metadata and edited geometry.
        var (imgDoc, tax) = serializer.DeserializeImage(File.ReadAllText(Path.Combine(outDir, "annotations", "specimen_001.json")));
        var exported = imgDoc.Annotations.Single();
        Assert.Equal(annotationId, exported.Id);
        Assert.Equal(VerificationStatus.VisuallyEstimated, exported.VerificationStatus);
        Assert.Equal("Dark red cuprite-like layer; optical only.", exported.Notes);
        Assert.Equal(2, exported.Revision);
        Assert.True(exported.Mask.Equals(editedMask));
        Assert.Equal("Cu2O", tax.Get(2).Name);

        // Audit trail distinguishes user actions from algorithm output.
        var log = new ProjectStore(serializer).OpenAuditLog(projectFolder).ReadAll();
        Assert.Contains(log, e => e.Action == "ROI Created" && e.Actor == "user:tester");
        Assert.Contains(log, e => e.Action == "GrabCut Segmentation" && e.Actor.StartsWith("algorithm:GrabCut@"));
        Assert.Contains(log, e => e.Action == "Foreground Correction");
        Assert.Contains(log, e => e.Action == "Segmentation Accepted");
        Assert.Contains(log, e => e.Action == "Class Assigned" && e.Details.Contains("VisuallyEstimated"));
        Assert.Contains(log, e => e.Action == "Polygon Edited");

        // SQLite index reflects the saved state.
        var index = new SqliteProjectIndex(Path.Combine(projectFolder, NativeFormat.IndexDatabaseName));
        var row = Assert.Single(index.Query(new AnnotationQuery { ClassId = 2 }));
        Assert.Equal(2, row.Revision);
        Assert.Equal(1, index.Count("revisions"));
    }
}
