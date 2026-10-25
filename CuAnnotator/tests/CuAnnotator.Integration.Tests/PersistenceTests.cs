using CuAnnotator.Application.Annotations;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Contracts;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Integration.Tests;

public sealed class PersistenceTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "cuann-persist-" + Guid.NewGuid().ToString("N"));
    private readonly NativeJsonSerializer _ser = new();

    public void Dispose() { try { Directory.Delete(_root, true); } catch (IOException) { } }

    private async Task<(AnnotationProject, string, ImageItem, Mat)> ProjectWithImage()
    {
        var folder = Path.Combine(_root, "p");
        var svc = new ProjectService(new ProjectStore(_ser));
        var p = svc.Create(new NewProjectRequest { Name = "t", Folder = folder });
        Directory.CreateDirectory(_root);
        var file = Path.Combine(_root, "img.png");
        using (var img = new Mat(300, 400, MatType.CV_8UC3, new Scalar(70, 120, 190)))
        {
            Cv2.Circle(img, new Point(120, 150), 60, new Scalar(40, 40, 150), -1);
            Cv2.Circle(img, new Point(120, 150), 20, new Scalar(70, 120, 190), -1);
            Cv2.Rectangle(img, new Rect(250, 50, 100, 200), new Scalar(110, 160, 60), -1);
            Cv2.ImWrite(file, img);
        }
        var r = await svc.ImportImagesAsync(p, folder, new[] { file });
        var li = ProjectService.LoadImage(folder, r.Imported[0], out _);
        return (p, folder, r.Imported[0], li.Bgr);
    }

    [Fact]
    public async Task NativeRoundTrip_SemanticMaskIsPixelIdentical_AndAllMetadataSurvives()
    {
        var (p, folder, item, bgr) = await ProjectWithImage();
        using var _ = bgr;
        var s = new AnnotationSession(p, item, bgr, new SegmentationEngine(), new MemoryAuditLog(), "u");
        s.BeginRoi(new PixelRect(50, 80, 140, 140), new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.LabColor });
        s.AddGuidance(GuidanceKind.Foreground, Stroke.Point(new PixelPoint(120, 110), 3)); // on the ring, not the hole
        await s.RunSegmentationAsync();
        var a1 = s.AcceptCandidate(2, VerificationStatus.ExpertReviewed, "ring with hole").Annotation;
        Assert.Single(a1.Polygons[0].Holes);
        s.BeginRoi(new PixelRect(240, 40, 120, 220), new SegmentationParameters());
        await s.RunSegmentationAsync();
        var a2 = s.AcceptCandidate(6, VerificationStatus.InstrumentallyVerified).Annotation;
        s.SetVerification(a2.Id, VerificationStatus.InstrumentallyVerified, "dr. x", new EvidenceReference { Type = EvidenceType.Raman, Source = "run-17", Notes = "peaks at 433 cm-1" });
        s.UpdateMetadata(a2.Id, "Severity", a => a.Severity = "Moderate");
        s.SetVisibility(a1.Id, false);
        var before = LabelMapComposer.Compose(item).Labels;

        await new ProjectStore(_ser).SaveAsync(p, folder);
        var back = await new ProjectStore(_ser).LoadAsync(folder);
        var img2 = back.Images.Single();
        Assert.Equal(before, LabelMapComposer.Compose(img2).Labels);
        var b2 = img2.Annotations.Single(a => a.Id == a2.Id);
        Assert.Equal("Moderate", b2.Severity);
        Assert.Equal(EvidenceType.Raman, b2.Evidence.Single().Type);
        Assert.Equal("dr. x", b2.ReviewedBy);
        Assert.False(img2.Annotations.Single(a => a.Id == a1.Id).IsVisible);
        Assert.Equal(_ser.Serialize(p).Split('\n').Where(l => !l.Contains("savedAt")), _ser.Serialize(back).Split('\n').Where(l => !l.Contains("savedAt")));
    }

    [Fact]
    public async Task Save_KeepsBackup_AndCorruptMainFallsBackToBackup()
    {
        var (p, folder, _, bgr) = await ProjectWithImage();
        bgr.Dispose();
        var store = new ProjectStore(_ser);
        await store.SaveAsync(p, folder);
        p.Description = "second";
        await store.SaveAsync(p, folder);
        Assert.True(File.Exists(Path.Combine(folder, NativeFormat.BackupFileName)));
        File.WriteAllText(Path.Combine(folder, NativeFormat.ProjectFileName), "{ truncated");
        var loaded = await store.LoadAsync(folder);
        Assert.NotNull(store.LastLoadWarning);
        Assert.Equal("t", loaded.Name);
    }

    [Fact]
    public async Task Autosave_RecoveryData_IsDetectedAndDiscardedAfterSave()
    {
        var (p, folder, _, bgr) = await ProjectWithImage();
        bgr.Dispose();
        var store = new ProjectStore(_ser);
        await store.SaveAsync(p, folder);
        await Task.Delay(20);
        p.Description = "unsaved work";
        await store.SaveRecoveryAsync(p, folder);
        Assert.True(store.HasRecoveryData(folder));
        Assert.Equal("unsaved work", (await store.LoadRecoveryAsync(folder))!.Description);
        await store.SaveAsync(p, folder);
        Assert.False(store.HasRecoveryData(folder));
    }

    [Fact]
    public void FutureSchemaVersion_IsRejected()
    {
        var json = _ser.Serialize(new AnnotationProject { Name = "x" }).Replace("\"schemaVersion\": 1", "\"schemaVersion\": 99");
        Assert.Throws<InvalidDataException>(() => _ser.Deserialize(json));
    }

    [Fact]
    public async Task UndoRedo_CoversBrushMergeSplitRelabelDelete()
    {
        var (p, _, item, bgr) = await ProjectWithImage();
        using var _b = bgr;
        var s = new AnnotationSession(p, item, bgr, new SegmentationEngine(), new MemoryAuditLog(), "u");
        var a = s.AcceptMask(new RasterMask(new PixelRect(10, 10, 20, 20), Enumerable.Repeat((byte)1, 400).ToArray()), 3, VerificationStatus.Unverified, "Manual").Annotation;
        var b = s.AcceptMask(new RasterMask(new PixelRect(40, 10, 20, 20), Enumerable.Repeat((byte)1, 400).ToArray()), 3, VerificationStatus.Unverified, "Manual").Annotation;
        var mapBefore = LabelMapComposer.Compose(item).Labels;

        s.BrushEdit(a.Id, new Stroke(new[] { new PixelPoint(30, 20), new PixelPoint(39, 20) }, 2), add: true);
        Assert.True(a.PixelArea > 400);
        s.Merge(new[] { a.Id, b.Id });
        Assert.Single(item.Annotations);
        s.Relabel(a.Id, 4);
        var parts = s.Split(a.Id, new[] { new PixelPoint(35, 0), new PixelPoint(35, 40) });
        Assert.Equal(2, parts.Count);
        s.Delete(new[] { parts[1].Id });
        Assert.Single(item.Annotations);

        for (int i = 0; i < 5; i++) s.Undo();
        Assert.Equal(2, item.Annotations.Count);
        Assert.Equal(mapBefore, LabelMapComposer.Compose(item).Labels);
        Assert.Same(a, item.Annotations[0]); // identity preserved for UI bindings
        for (int i = 0; i < 5; i++) s.Redo();
        Assert.Single(item.Annotations);
        Assert.Equal(4, item.Annotations[0].ClassId);
    }

    [Theory]
    [InlineData(OverlapPolicy.PreserveExisting)]
    [InlineData(OverlapPolicy.ReplaceExisting)]
    [InlineData(OverlapPolicy.AllowOverlapTopmostWins)]
    public async Task OverlapPolicies_AreDeterministic(OverlapPolicy policy)
    {
        var (p, _, item, bgr) = await ProjectWithImage();
        using var _b = bgr;
        var s = new AnnotationSession(p, item, bgr, new SegmentationEngine(), new MemoryAuditLog(), "u");
        var sq = Enumerable.Repeat((byte)1, 400).ToArray();
        var first = s.AcceptMask(new RasterMask(new PixelRect(0, 0, 20, 20), sq), 2, VerificationStatus.Unverified, "M").Annotation;
        var r = s.AcceptMask(new RasterMask(new PixelRect(10, 0, 20, 20), (byte[])sq.Clone()), 3, VerificationStatus.Unverified, "M", policy: policy);
        Assert.Equal(200, r.OverlapPixels);
        var map = LabelMapComposer.Compose(item);
        switch (policy)
        {
            case OverlapPolicy.PreserveExisting:
                Assert.Equal(400, first.PixelArea); Assert.Equal(200, r.Annotation.PixelArea); Assert.Equal(0, map.OverlapPixels); break;
            case OverlapPolicy.ReplaceExisting:
                Assert.Equal(200, first.PixelArea); Assert.Equal(400, r.Annotation.PixelArea); Assert.Equal(0, map.OverlapPixels); break;
            default:
                Assert.Equal(200, map.OverlapPixels); Assert.Equal(3, map.Labels[15]); break;
        }
    }

    [Fact]
    public async Task StaleSegmentationResult_IsDiscarded_WhenCandidateChangesDuringRun()
    {
        var (p, _, item, bgr) = await ProjectWithImage();
        using var _b = bgr;
        var s = new AnnotationSession(p, item, bgr, new SegmentationEngine(), new MemoryAuditLog(), "u");
        s.BeginRoi(new PixelRect(50, 80, 140, 140), new SegmentationParameters());
        var run = s.RunSegmentationAsync();
        s.AddGuidance(GuidanceKind.Foreground, Stroke.Point(new PixelPoint(120, 110), 3)); // user keeps working
        Assert.Null(await run);
        Assert.Null(s.Candidate!.Result);
        Assert.Equal(2, s.History.Count); // ROI + stroke only — no phantom result step
    }

    [Fact]
    public async Task ClassSuggester_SuggestsSimilarClass_WithoutChangingVerification()
    {
        var (p, _, item, bgr) = await ProjectWithImage();
        using var _b = bgr;
        var s = new AnnotationSession(p, item, bgr, new SegmentationEngine(), new MemoryAuditLog(), "u");
        s.BeginRoi(new PixelRect(240, 40, 120, 220), new SegmentationParameters());
        await s.RunSegmentationAsync();
        var patina = s.AcceptCandidate(6, VerificationStatus.ExpertReviewed).Annotation;
        ClassSuggester.ComputeDescriptor(bgr, patina);
        s.BeginRoi(new PixelRect(50, 80, 140, 140), new SegmentationParameters());
        await s.RunSegmentationAsync();
        var oxide = s.AcceptCandidate(2, VerificationStatus.VisuallyEstimated).Annotation;
        ClassSuggester.ComputeDescriptor(bgr, oxide);

        var probe = new RasterMask(new PixelRect(270, 100, 30, 30), Enumerable.Repeat((byte)1, 900).ToArray());
        var d = Vision.ColorAnalysis.RegionStatistics.Compute(bgr, probe).ToDescriptor();
        var sug = new ClassSuggester { K = 1 }.Suggest(p, d)!;
        Assert.Equal(6, sug.ClassId);
        Assert.InRange(sug.SimilarityScore, 0, 1);
    }
}
