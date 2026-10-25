using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Desktop.Interaction;
using CuAnnotator.Desktop.Rendering;
using CuAnnotator.Desktop.Services;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.Geometry;
using CuAnnotator.Vision.ImageIO;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;
using Serilog;

namespace CuAnnotator.Desktop.ViewModels;

/// <summary>Editing state for the image currently shown in the workspace.</summary>
public sealed partial class EditorViewModel : ObservableObject, IDisposable
{
    private readonly MainViewModel _main;
    private readonly SegmentationEngine _engine;
    private readonly IDialogService _dialogs;
    private readonly UserSettings _settings;
    private readonly LoadedImage _loaded;
    private readonly Dictionary<Guid, (RasterMask Mask, int ClassId, bool Visible, string Color, AnnotationVisual Visual)> _visualCache = new();
    private CancellationTokenSource? _cts;
    private readonly EventHandler _historyHandler;

    public AnnotationSession Session { get; }
    public ViewportScene Scene { get; } = new();
    public TiledImageSource Tiles { get; }
    public ImageItem Image => Session.Image;
    public Mat Source => _loaded.Bgr;
    public ObservableCollection<AnnotationViewModel> Annotations { get; } = new();
    public ObservableCollection<RegionCandidate> RegionCandidates { get; } = new();
    public HashSet<Guid> SelectedIds => Scene.Selected;

    [ObservableProperty] private AnnotationViewModel? _selectedAnnotation;
    [ObservableProperty] private bool _isSegmenting;
    [ObservableProperty] private string _segmentationStatus = "Draw an ROI around a corrosion region.";
    [ObservableProperty] private string _pixelInfo = "";
    [ObservableProperty] private string _cursorText = "";
    [ObservableProperty] private string _stagesText = "";
    [ObservableProperty] private bool _hasCandidate;

    public EditorViewModel(MainViewModel main, AnnotationSession session, LoadedImage loaded, SegmentationEngine engine, IDialogService dialogs, UserSettings settings)
    {
        _main = main; Session = session; _loaded = loaded; _engine = engine; _dialogs = dialogs; _settings = settings;
        Tiles = new TiledImageSource(loaded.Bgr);
        Scene.Image = Tiles;
        Scene.Opacity = settings.OverlayOpacity;
        session.AnnotationsChanged += (_, _) => { SyncAnnotations(); _main.MarkDirty(); };
        session.CandidateChanged += (_, _) => RefreshCandidate();
        _historyHandler = (_, _) => _main.RefreshUndoState();
        session.History.Changed += _historyHandler;
        SyncAnnotations();
        RefreshCandidate();
    }

    public SegmentationParameters Parameters => _main.Parameters;
    private int BrushRadius => _main.BrushRadius;

    // ───────────── Scene synchronisation ─────────────

    public void SyncAnnotations()
    {
        var tax = Session.Project.Taxonomy;
        var existing = Annotations.ToDictionary(a => a.Id);
        var selectedId = SelectedAnnotation?.Id;
        _syncing = true;
        Annotations.Clear();
        foreach (var a in Image.Annotations.OrderBy(a => a.ZOrder))
        {
            var vm = existing.TryGetValue(a.Id, out var e) ? e : new AnnotationViewModel(a, Session, tax);
            vm.Refresh();
            Annotations.Add(vm);
        }
        _syncing = false;
        SelectedIds.RemoveWhere(id => Image.Annotations.All(a => a.Id != id));
        SelectedAnnotation = Annotations.FirstOrDefault(a => a.Id == selectedId && SelectedIds.Contains(a.Id))
                             ?? (SelectedIds.Count > 0 ? Annotations.FirstOrDefault(a => a.Id == SelectedIds.Last()) : null);
        SelectedAnnotation?.Refresh();
        NotifySelectionCommands();
        RebuildVisuals();
    }

    public void RebuildVisuals()
    {
        var tax = Session.Project.Taxonomy;
        Scene.Annotations.Clear();
        var alive = new HashSet<Guid>();
        foreach (var a in Image.Annotations.OrderBy(a => a.ZOrder))
        {
            alive.Add(a.Id);
            var color = tax.Find(a.ClassId)?.ColorHex ?? "";
            if (!_visualCache.TryGetValue(a.Id, out var c) || !ReferenceEquals(c.Mask, a.Mask) || c.ClassId != a.ClassId || c.Visible != a.IsVisible || c.Color != color)
            {
                c = (a.Mask, a.ClassId, a.IsVisible, color, OverlayFactory.Build(a, tax));
                _visualCache[a.Id] = c;
            }
            Scene.Annotations.Add(c.Visual);
        }
        foreach (var dead in _visualCache.Keys.Where(k => !alive.Contains(k)).ToList()) _visualCache.Remove(dead);
        Scene.Invalidate();
    }

    private void RefreshCandidate()
    {
        var c = Session.Candidate;
        IsSegmenting = Session.IsSegmenting;
        HasCandidate = c?.Result is not null;
        Scene.CandidateRoi = c?.Roi;
        Scene.Candidate = null;
        Scene.Guidance.Clear();
        if (c is not null)
        {
            if (c.Result is { } r)
            {
                var (bmp, rect) = OverlayFactory.MaskBitmap(r.Mask, Colors.Cyan);
                Scene.Candidate = new AnnotationVisual
                {
                    Id = Guid.Empty, Color = Colors.Cyan, Fill = bmp, FillRect = rect, Outline = OverlayFactory.Outline(r.Components),
                    Visible = true, ClassId = 0,
                };
                StagesText = string.Join("\n", r.Stages.Select(s => $"{s.Name}: {s.Mask.PixelCount:N0} px — {s.Description}"))
                             + (r.Warnings.Count > 0 ? "\n⚠ " + string.Join("\n⚠ ", r.Warnings) : "");
            }
            else StagesText = "";
            AddStrokes(c.Positive, Color.FromArgb(170, 40, 200, 80));
            AddStrokes(c.Negative, Color.FromArgb(170, 230, 50, 50));
            AddStrokes(c.Barrier, Color.FromArgb(220, 255, 200, 0));
        }
        else StagesText = "";
        Scene.Invalidate();
        AcceptCandidateCommand.NotifyCanExecuteChanged();
    }

    private void AddStrokes(IEnumerable<Stroke> strokes, Color color)
    {
        foreach (var s in strokes)
            Scene.Guidance.Add(new PreviewShape
            {
                Points = s.Points.Select(ImagePoint.CenterOf).ToList(), Color = color, ImageThickness = Math.Max(1, 2 * s.Radius + 1),
            });
    }

    // ───────────── Segmentation ─────────────

    public void BeginRoi(PixelRect roi, SegmentationAlgorithmKind? algorithm = null)
    {
        var p = Parameters.Clone();
        if (algorithm is { } k) p.Algorithm = k;
        try { Session.BeginRoi(roi, p); }
        catch (ArgumentException ex) { SegmentationStatus = ex.Message; return; }
        SegmentationStatus = $"ROI {roi.Width}×{roi.Height} px. Running {p.Algorithm}…";
        _ = RunSegmentationAsync();
    }

    public void BeginRegion(IReadOnlyList<PixelPoint> outline, string kind)
    {
        if (outline.Count < 3) { SegmentationStatus = "Draw at least three points."; return; }
        var region = MaskRasterizer.Rasterize(new[] { new PolygonComponent(outline) }, Image.Width, Image.Height);
        try { Session.BeginRegion(region, Parameters, kind); }
        catch (ArgumentException ex) { SegmentationStatus = ex.Message; return; }
        _ = RunSegmentationAsync();
    }

    public void AddGuidance(GuidanceKind kind, Stroke stroke)
    {
        if (Session.Candidate is null) { SegmentationStatus = "Draw an ROI first; brushes then refine the candidate."; return; }
        Session.AddGuidance(kind, stroke);
        if (_main.LivePreview) _ = RunSegmentationAsync();
        else SegmentationStatus = "Guidance added. Press G (Refine) to update the segmentation.";
    }

    [RelayCommand]
    public async Task RunSegmentationAsync()
    {
        if (Session.Candidate is null) { SegmentationStatus = "No ROI selected."; return; }
        _cts?.Cancel();
        _cts = new CancellationTokenSource();
        var token = _cts.Token;
        Session.UpdateParameters(Parameters);
        try
        {
            SegmentationStatus = $"Running {Parameters.Algorithm}…";
            var r = await Track(Session.RunSegmentationAsync(token));
            if (r is null) return; // superseded by newer input
            SegmentationStatus = $"{r.AlgorithmName}: {r.Mask.PixelCount:N0} px, {r.Components.Count} component(s), " +
                                 $"{r.Components.Sum(c => c.Holes.Count)} hole(s) in {r.ProcessingTime.TotalMilliseconds:0} ms. " +
                                 "Refine with brushes, then Enter to accept.";
            ComputeSuggestionForCandidate(r);
        }
        catch (OperationCanceledException) { SegmentationStatus = "Segmentation cancelled."; }
        catch (SegmentationException ex) { SegmentationStatus = ex.Message; }
        catch (OpenCVException ex)
        {
            Log.Error(ex, "OpenCV failure during segmentation");
            SegmentationStatus = $"OpenCV error: {ex.Message}";
        }
    }

    [ObservableProperty] private string _candidateSuggestion = "";
    private ClassSuggestionResult? _candidateSuggestionResult;

    private void ComputeSuggestionForCandidate(SegmentationResult r)
    {
        CandidateSuggestion = "";
        _candidateSuggestionResult = null;
        if (!_main.ClassAssistEnabled || r.Mask.IsEmpty) return;
        var d = RegionStatistics.Compute(Source, r.Mask).ToDescriptor();
        var s = new ClassSuggester().Suggest(Session.Project, d);
        if (s is null) return;
        _candidateSuggestionResult = s;
        CandidateSuggestion = $"Suggested class: {s.ClassName} — similarity {s.SimilarityScore:0.00} (uncalibrated), {s.VoteShare:P0} of neighbour weight.\nBasis: {s.Basis}.";
    }

    [RelayCommand]
    private void CancelSegmentation()
    {
        _cts?.Cancel();
        if (!IsSegmenting) { Session.CancelCandidate(); SegmentationStatus = "Candidate discarded."; }
    }

    [RelayCommand] private void ClearGuidance() => Session.ClearGuidance();

    [RelayCommand(CanExecute = nameof(HasCandidate))]
    private void AcceptCandidate() => AcceptWithClass(_main.ActiveClass?.Id ?? 2);

    public void AcceptWithClass(int classId)
    {
        if (Session.Candidate?.Result is null) return;
        var policy = Session.Project.Settings.OverlapPolicy;
        if (policy == OverlapPolicy.AskUser)
        {
            long ov = Session.OverlapWithExisting(Session.Candidate.Result.Mask, Session.Candidate.EditingAnnotationId);
            if (ov > 0)
            {
                var chosen = _dialogs.AskOverlapPolicy(ov);
                if (chosen is null) return;
                policy = chosen.Value;
            }
            else policy = OverlapPolicy.PreserveExisting;
        }
        try
        {
            var res = Session.AcceptCandidate(classId, _main.DefaultVerification, policy: policy);
            ClassSuggester.ComputeDescriptor(Source, res.Annotation);
            if (_candidateSuggestionResult is { } s)
                res.Annotation.Suggestion = new ClassSuggestion { ClassId = s.ClassId, SimilarityScore = s.SimilarityScore, Basis = s.Basis, CreatedAt = DateTimeOffset.UtcNow };
            CandidateSuggestion = "";
            Select(res.Annotation.Id, additive: false);
            var cls = Session.Project.Taxonomy.Find(classId)?.Name;
            SegmentationStatus = $"Accepted as {cls} ({res.Annotation.PixelArea:N0} px)" +
                                 (res.OverlapPixels > 0 ? $"; {res.OverlapPixels:N0} overlapping px resolved by {policy}" : "") + ".";
            _main.FlashMessage($"✔ {cls} assigned");
        }
        catch (InvalidOperationException ex) { _dialogs.ShowError(ex.Message); }
    }

    /// <summary>Number key / palette click: accept the candidate with this class, or relabel the selection.</summary>
    public void AssignClass(int classId)
    {
        if (Session.Candidate?.Result is not null) { AcceptWithClass(classId); return; }
        if (SelectedIds.Count == 0) { SegmentationStatus = "Select an annotation or create a candidate before assigning a class."; return; }
        foreach (var id in SelectedIds.ToList()) Session.Relabel(id, classId);
        _main.FlashMessage($"✔ {Session.Project.Taxonomy.Find(classId)?.Name} assigned to {SelectedIds.Count} annotation(s)");
    }

    // ───────────── Selection & editing ─────────────

    public Annotation? HitTest(PixelPoint p) =>
        Image.Annotations.Where(a => a.IsVisible && a.Mask.Get(p.X, p.Y)).OrderByDescending(a => a.ZOrder).FirstOrDefault();

    public void Select(Guid? id, bool additive)
    {
        if (!additive) SelectedIds.Clear();
        if (id is { } g)
        {
            if (additive && SelectedIds.Contains(g)) SelectedIds.Remove(g); else SelectedIds.Add(g);
        }
        SelectedAnnotation = SelectedIds.Count == 0 ? null : Annotations.FirstOrDefault(a => a.Id == SelectedIds.Last());
        Scene.Invalidate();
        NotifySelectionCommands();
    }

    private bool _syncing;

    partial void OnSelectedAnnotationChanged(AnnotationViewModel? value)
    {
        if (_syncing) return; // transient null while the list is rebuilt
        if (value is not null && !SelectedIds.Contains(value.Id)) { SelectedIds.Clear(); SelectedIds.Add(value.Id); Scene.Invalidate(); }
        NotifySelectionCommands();
    }

    private void NotifySelectionCommands()
    {
        DeleteSelectedCommand.NotifyCanExecuteChanged();
        MergeSelectedCommand.NotifyCanExecuteChanged();
        FillHolesCommand.NotifyCanExecuteChanged();
        DuplicateSelectedCommand.NotifyCanExecuteChanged();
        ResegmentSelectedCommand.NotifyCanExecuteChanged();
        UnionCommand.NotifyCanExecuteChanged();
        IntersectCommand.NotifyCanExecuteChanged();
        SubtractCommand.NotifyCanExecuteChanged();
        AcceptSuggestionCommand.NotifyCanExecuteChanged();
    }

    private bool HasSelection() => SelectedIds.Count > 0;
    private bool HasSingleSelection() => SelectedIds.Count == 1;
    private bool HasTwoSelected() => SelectedIds.Count == 2;
    private bool HasMultiSelection() => SelectedIds.Count >= 2;

    [RelayCommand(CanExecute = nameof(HasSelection))]
    private void DeleteSelected()
    {
        if (Session.Candidate is not null && SelectedIds.Count == 0) { Session.CancelCandidate(); return; }
        Session.Delete(SelectedIds.ToList());
        Select(null, false);
    }

    [RelayCommand(CanExecute = nameof(HasMultiSelection))]
    private void MergeSelected()
    {
        try
        {
            var ids = SelectedIds.ToList();
            bool force = false;
            if (ids.Select(i => Session.Find(i).ClassId).Distinct().Count() > 1)
            {
                if (!_dialogs.Confirm("The selected regions have different classes. Merge into the class of the first selected region?")) return;
                force = true;
            }
            var a = Session.Merge(ids, force);
            Select(a.Id, false);
        }
        catch (InvalidOperationException ex) { _dialogs.ShowError(ex.Message); }
    }

    [RelayCommand(CanExecute = nameof(HasSingleSelection))]
    private void FillHoles() => Guard(() => Session.FillHoles(SelectedIds.Single()));

    [RelayCommand(CanExecute = nameof(HasSingleSelection))]
    private void DuplicateSelected() => Guard(() => Select(Session.Duplicate(SelectedIds.Single()).Id, false));

    [RelayCommand(CanExecute = nameof(HasSingleSelection))]
    private void ResegmentSelected() => Guard(() =>
    {
        Session.BeginEditExisting(SelectedIds.Single(), Parameters);
        _ = RunSegmentationAsync();
    });

    [RelayCommand(CanExecute = nameof(HasTwoSelected))] private void Union() => BooleanOp(AnnotationSession.BooleanOp.Union);
    [RelayCommand(CanExecute = nameof(HasTwoSelected))] private void Intersect() => BooleanOp(AnnotationSession.BooleanOp.Intersection);
    [RelayCommand(CanExecute = nameof(HasTwoSelected))] private void Subtract() => BooleanOp(AnnotationSession.BooleanOp.Subtraction);

    private void BooleanOp(AnnotationSession.BooleanOp op)
    {
        // Target = the selected annotation shown in the properties panel; operand = the other one.
        var target = SelectedAnnotation?.Id ?? SelectedIds.First();
        var operand = SelectedIds.First(i => i != target);
        Guard(() => Session.Boolean(target, operand, op));
    }

    [RelayCommand(CanExecute = nameof(CanAcceptSuggestion))]
    private void AcceptSuggestion() => Guard(() => Session.AcceptSuggestion(SelectedIds.Single()));
    private bool CanAcceptSuggestion() => SelectedIds.Count == 1 && Image.Annotations.FirstOrDefault(a => a.Id == SelectedIds.First())?.Suggestion is not null;

    [RelayCommand]
    private void RestoreRevision(AnnotationRevision? r)
    {
        if (r is null || SelectedAnnotation is null) return;
        Guard(() => Session.RestoreRevision(SelectedAnnotation.Id, r.Revision));
    }

    [RelayCommand]
    private void AddEvidence()
    {
        if (SelectedAnnotation is null) return;
        var ev = _dialogs.AskEvidence();
        if (ev is null) return;
        var status = ev.Value.Status;
        Guard(() => Session.SetVerification(SelectedAnnotation.Id, status, _settings.UserName, ev.Value.Evidence));
    }

    public void BrushOnSelection(Stroke stroke, bool add)
    {
        if (SelectedIds.Count != 1) { SegmentationStatus = "Select one annotation to paint on it, or draw an ROI to create a candidate."; return; }
        Guard(() => Session.BrushEdit(SelectedIds.Single(), stroke, add));
    }

    public void Split(IReadOnlyList<PixelPoint> cut)
    {
        if (SelectedIds.Count != 1) { SegmentationStatus = "Select the annotation to split, then draw the cut line."; return; }
        Guard(() =>
        {
            var parts = Session.Split(SelectedIds.Single(), cut);
            SegmentationStatus = $"Split into {parts.Count} annotations.";
        });
    }

    public void FillAt(PixelPoint p)
    {
        var target = SelectedIds.Count == 1 ? Session.Find(SelectedIds.Single()) : null;
        target ??= Image.Annotations.Where(a => a.IsVisible && a.Mask.Bounds.Contains(p)).OrderByDescending(a => a.ZOrder).FirstOrDefault();
        if (target is null) { SegmentationStatus = "Click inside a hole of an annotation."; return; }
        Guard(() => Session.FillHoles(target.Id));
    }

    public void ApplyPolygonEdit(Guid id, List<PolygonComponent> polygons) => Guard(() => Session.EditPolygons(id, polygons));

    public void Move(Guid id, int dx, int dy) => Guard(() => Session.Move(id, dx, dy));

    public async Task MagicWandAsync(PixelPoint p)
    {
        try
        {
            SegmentationStatus = "Magic wand…";
            var r = await Track(_engine.MagicWandAsync(Source, p, Parameters, 1024, CancellationToken.None));
            Session.ProposeCandidate(r, "Magic Wand Selection");
            SegmentationStatus = $"Magic wand: {r.Mask.PixelCount:N0} px (ΔE ≤ {Parameters.ColorThreshold}). Enter to accept.";
            ComputeSuggestionForCandidate(r);
        }
        catch (Exception ex) when (ex is SegmentationException or InvalidOperationException) { SegmentationStatus = ex.Message; }
    }

    public async Task ColorSelectAsync(PixelPoint p, PixelRect? within)
    {
        try
        {
            var roi = within ?? new PixelRect(0, 0, Image.Width, Image.Height);
            SegmentationStatus = "Colour selection…";
            var r = await Track(_engine.ColorSelectAsync(Source, roi, new[] { Stroke.Point(p, 2) }, Parameters, CancellationToken.None));
            Session.ProposeCandidate(r, "Colour-Based Selection");
            SegmentationStatus = $"Colour selection: {r.Mask.PixelCount:N0} px in {r.Components.Count} region(s). Enter to accept.";
        }
        catch (Exception ex) when (ex is SegmentationException or InvalidOperationException) { SegmentationStatus = ex.Message; }
    }

    public List<PixelPoint> SnapBoundary(IReadOnlyList<PixelPoint> path) => BoundarySnapper.Snap(Source, path, _settings.SnapRadius);

    // ───────────── Research extensions ─────────────

    [RelayCommand]
    private async Task FindRegionSuggestionsAsync()
    {
        SegmentationStatus = "Searching unlabeled regions…";
        var refs = _main.BatchReferences;
        var list = await Track(Task.Run(() => RegionSuggester.Suggest(Source, Image, Parameters.ColorThreshold, Math.Max(20, Parameters.MinComponentArea), 50, refs)));
        RegionCandidates.Clear();
        foreach (var c in list) RegionCandidates.Add(c);
        SegmentationStatus = list.Count == 0 ? "No candidate regions found." : $"{list.Count} candidate region(s) — select one to preview, then accept or skip.";
    }

    [RelayCommand]
    private void PreviewRegionCandidate(RegionCandidate? c)
    {
        if (c is null) return;
        Session.ProposeCandidate(new SegmentationResult
        {
            AlgorithmName = "Region Suggestion", AlgorithmVersion = "1.0.0", Mask = c.Mask,
            Components = Vision.Contours.ContourExtractor.Extract(c.Mask), ProcessingTime = TimeSpan.Zero,
            ParametersUsed = new() { ["Algorithm"] = "LabColor", ["ColorThreshold"] = Parameters.ColorThreshold.ToString(CultureInfo.InvariantCulture), ["suggestedClass"] = c.SuggestedClassId.ToString() },
        }, "Region Suggestion");
        CandidateSuggestion = $"Candidate region — {c.Reason}. This is a visual proposal; assign the class yourself.";
        RegionCandidates.Remove(c);
    }

    [RelayCommand]
    private void ExportRegionStatistics()
    {
        var path = _dialogs.SaveFile("CSV (*.csv)|*.csv", Path.GetFileNameWithoutExtension(Image.FileName) + "_region_statistics.csv");
        if (path is null) return;
        File.WriteAllText(path, RegionStatisticsCsv.Build(Session.Project, Image, Source));
        SegmentationStatus = $"Region statistics written to {path}.";
    }

    // ───────────── Pixel inspector ─────────────

    public void UpdateCursor(ImagePoint p)
    {
        var px = p.ToPixel();
        bool inside = px.X >= 0 && px.Y >= 0 && px.X < Image.Width && px.Y < Image.Height;
        CursorText = inside ? $"x {px.X}, y {px.Y}" : "";
        if (!inside) return;
        var v = Source.At<Vec3b>(px.Y, px.X);
        byte b = v.Item0, g = v.Item1, r = v.Item2;
        var hsv = ColorScience.RgbToHsv(r, g, b);
        var lab = ColorScience.RgbToLab(r, g, b);
        var label = Image.Annotations.Where(a => a.Mask.Get(px.X, px.Y)).OrderByDescending(a => a.ZOrder).FirstOrDefault();
        string um = Image.MicrometersPerPixel is { } mpp ? $"  = ({px.X * mpp:0.##}, {px.Y * mpp:0.##}) µm" : "";
        PixelInfo = $"Pixel Coordinate : ({px.X}, {px.Y}){um}\n\nRGB:\nR = {r}\nG = {g}\nB = {b}\n\n" +
                    $"HSV:\nH = {hsv.H:0.0}°\nS = {hsv.S:0.000}\nV = {hsv.V:0.000}\n\n" +
                    $"CIELAB (D65, sRGB assumed):\nL* = {lab.L:0.00}\na* = {lab.A:0.00}\nb* = {lab.B:0.00}\n\n" +
                    $"Gray = {ColorScience.Gray(r, g, b):0.0}\nLabel: {(label is null ? "0 (unlabeled)" : $"{label.ClassId} ({Session.Project.Taxonomy.Find(label.ClassId)?.Name})")}";
    }

    private void Guard(Action a)
    {
        try { a(); }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or KeyNotFoundException) { _dialogs.ShowError(ex.Message); }
    }

    private readonly List<Task> _background = new();

    /// <summary>Tracks work that reads <see cref="Source"/> so the native image is not freed underneath it.</summary>
    private T Track<T>(T task) where T : Task
    {
        lock (_background) { _background.RemoveAll(t => t.IsCompleted); _background.Add(task); }
        return task;
    }

    public void Dispose()
    {
        _cts?.Cancel();
        Session.History.Changed -= _historyHandler;
        Task[] pending;
        lock (_background) pending = _background.Where(t => !t.IsCompleted).ToArray();
        if (pending.Length == 0) { Tiles.Dispose(); _loaded.Dispose(); return; }
        Task.WhenAll(pending).ContinueWith(_ => { Tiles.Dispose(); _loaded.Dispose(); }, TaskScheduler.Default);
    }
}
