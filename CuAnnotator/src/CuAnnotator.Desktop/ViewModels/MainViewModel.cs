using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows.Data;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Application.Commands;
using CuAnnotator.Application.Evaluation;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Application.Services;
using CuAnnotator.Contracts;
using CuAnnotator.Desktop.Interaction;
using CuAnnotator.Desktop.Rendering;
using CuAnnotator.Desktop.Services;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.ImageIO;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;
using Serilog;

namespace CuAnnotator.Desktop.ViewModels;

public enum ViewCommand { ZoomIn, ZoomOut, Fit, ActualPixels }

public sealed partial class MainViewModel : ObservableObject
{
    private readonly ProjectService _projects;
    private readonly ProjectStore _store;
    private readonly SegmentationEngine _engine;
    private readonly DatasetExporter _exporter;
    private readonly IDialogService _dialogs;
    private readonly UserSettings _settings;
    private readonly ThemeService _theme;
    private readonly ILogger _log;
    private readonly DispatcherTimer _autosave = new();
    private readonly DispatcherTimer _flashTimer = new() { Interval = TimeSpan.FromSeconds(2.5) };
    private readonly Dictionary<Guid, UndoRedoStack> _histories = new();
    private IAuditLog _audit = new MemoryAuditLog();

    public event EventHandler<ViewCommand>? ViewCommandRequested;

    public MainViewModel(ProjectService projects, ProjectStore store, SegmentationEngine engine, DatasetExporter exporter,
        IDialogService dialogs, UserSettings settings, ThemeService theme, ILogger log)
    {
        _projects = projects; _store = store; _engine = engine; _exporter = exporter; _dialogs = dialogs; _settings = settings; _theme = theme; _log = log;
        ClassesView = CollectionViewSource.GetDefaultView(Classes);
        ClassesView.Filter = o => o is ClassItemViewModel c && (string.IsNullOrWhiteSpace(ClassFilter)
            || c.Name.Contains(ClassFilter, StringComparison.OrdinalIgnoreCase)
            || (c.Model.ChemicalFormula?.Contains(ClassFilter, StringComparison.OrdinalIgnoreCase) ?? false)
            || (c.Parent?.Contains(ClassFilter, StringComparison.OrdinalIgnoreCase) ?? false));
        _brushRadius = settings.BrushRadius;
        _livePreview = settings.LivePreview;
        _overlayOpacity = settings.OverlayOpacity;
        _autosave.Tick += async (_, _) => await AutosaveAsync();
        _flashTimer.Tick += (_, _) => { FlashText = ""; _flashTimer.Stop(); };
        LoadTaxonomy(Taxonomy.CreateDefault());
    }

    // ───────────── State ─────────────

    [ObservableProperty] private AnnotationProject? _project;
    [ObservableProperty] private string? _projectFolder;
    [ObservableProperty] private EditorViewModel? _editor;
    [ObservableProperty] private ImageItemViewModel? _selectedImage;
    [ObservableProperty] private bool _isDirty;
    [ObservableProperty] private string _statusMessage = "Create or open a project to begin.";
    [ObservableProperty] private string _autosaveStatus = "";
    [ObservableProperty] private string _flashText = "";
    [ObservableProperty] private string _zoomText = "";
    [ObservableProperty] private bool _isBusy;
    [ObservableProperty] private double _progress;

    public ObservableCollection<ImageItemViewModel> Images { get; } = new();
    public ObservableCollection<ClassItemViewModel> Classes { get; } = new();
    public ICollectionView ClassesView { get; }
    [ObservableProperty] private string _classFilter = "";
    partial void OnClassFilterChanged(string value) => ClassesView.Refresh();
    [ObservableProperty] private ClassItemViewModel? _activeClass;

    public SegmentationParameters Parameters { get; } = new();
    public IReadOnlyList<SegmentationAlgorithmKind> Algorithms { get; } = new[]
    {
        SegmentationAlgorithmKind.GrabCut, SegmentationAlgorithmKind.LabColor, SegmentationAlgorithmKind.Watershed,
        SegmentationAlgorithmKind.ActiveContour, SegmentationAlgorithmKind.Hybrid,
    };
    public SegmentationAlgorithmKind SelectedAlgorithm
    {
        get => Parameters.Algorithm;
        set { Parameters.Algorithm = value; OnPropertyChanged(); OnPropertyChanged(nameof(Parameters)); }
    }
    public IReadOnlyList<ProcessingQuality> Qualities { get; } = Enum.GetValues<ProcessingQuality>();
    public IReadOnlyList<ColorDifferenceMetric> Metrics { get; } = Enum.GetValues<ColorDifferenceMetric>();
    public IReadOnlyList<ActiveContourMode> ContourModes { get; } = Enum.GetValues<ActiveContourMode>();
    public IReadOnlyList<VerificationStatus> VerificationStatuses { get; } = Enum.GetValues<VerificationStatus>();
    public IReadOnlyList<OverlayMode> OverlayModes { get; } = Enum.GetValues<OverlayMode>();
    public IReadOnlyList<ImageWorkflowState> WorkflowStates { get; } = Enum.GetValues<ImageWorkflowState>();
    public IReadOnlyList<string> SeverityLevels => Project?.Settings.SeverityLevels ?? new List<string> { "Mild", "Moderate", "Severe" };

    [ObservableProperty] private ToolKind _activeTool = ToolKind.Rectangle;
    [ObservableProperty] private int _brushRadius;
    [ObservableProperty] private bool _livePreview;
    [ObservableProperty] private bool _classAssistEnabled = true;
    [ObservableProperty] private VerificationStatus _defaultVerification = VerificationStatus.Unverified;
    [ObservableProperty] private OverlayMode _overlayMode = OverlayMode.Normal;
    [ObservableProperty] private double _overlayOpacity;
    [ObservableProperty] private bool _showBoundaries = true;
    [ObservableProperty] private bool _showMasks = true;
    [ObservableProperty] private bool _showGrid = true;
    public Dictionary<int, LabColor>? BatchReferences { get; private set; }
    [ObservableProperty] private string _batchReferenceText = "";

    partial void OnBrushRadiusChanged(int value) { _settings.BrushRadius = Math.Clamp(value, 0, 200); _settings.Save(); }
    partial void OnLivePreviewChanged(bool value) { _settings.LivePreview = value; _settings.Save(); }
    partial void OnOverlayModeChanged(OverlayMode value) => ApplySceneOptions();
    partial void OnOverlayOpacityChanged(double value) { _settings.OverlayOpacity = value; ApplySceneOptions(); }
    partial void OnShowBoundariesChanged(bool value) => ApplySceneOptions();
    partial void OnShowMasksChanged(bool value) => ApplySceneOptions();
    partial void OnShowGridChanged(bool value) => ApplySceneOptions();
    partial void OnActiveToolChanged(ToolKind value) => StatusMessage = ToolHints.For(value);

    private void ApplySceneOptions()
    {
        if (Editor is null) return;
        var s = Editor.Scene;
        s.Mode = OverlayMode; s.Opacity = OverlayOpacity; s.ShowBoundaries = ShowBoundaries; s.ShowMasks = ShowMasks; s.ShowGrid = ShowGrid;
        s.Invalidate();
    }

    public string Title => Project is null ? "CuAnnotator — Copper Corrosion Annotation Studio"
        : $"{Project.Name}{(IsDirty ? " •" : "")} — CuAnnotator";
    partial void OnIsDirtyChanged(bool value) => OnPropertyChanged(nameof(Title));
    partial void OnProjectChanged(AnnotationProject? value) { OnPropertyChanged(nameof(Title)); OnPropertyChanged(nameof(HasProject)); OnPropertyChanged(nameof(SeverityLevels)); }
    public bool HasProject => Project is not null;
    public string ProgressText => Project is null ? "" :
        $"{Images.Count(i => i.Model.Annotations.Count > 0)}/{Images.Count} images annotated · {Project.Images.Sum(i => i.Annotations.Count)} annotations";
    public string ImageIndexText => SelectedImage is null ? "" : $"Image {Images.IndexOf(SelectedImage) + 1} of {Images.Count}";

    public void MarkDirty()
    {
        IsDirty = true;
        SelectedImage?.Refresh();
        OnPropertyChanged(nameof(ProgressText));
    }

    public void RefreshUndoState()
    {
        UndoCommand.NotifyCanExecuteChanged();
        RedoCommand.NotifyCanExecuteChanged();
    }

    public void FlashMessage(string text)
    {
        FlashText = text;
        _flashTimer.Stop(); _flashTimer.Start();
    }

    private void LoadTaxonomy(Taxonomy t)
    {
        var active = ActiveClass?.Id;
        Classes.Clear();
        foreach (var c in t.Classes.Where(c => c.Id != Taxonomy.UnlabeledId).OrderBy(c => c.Id)) Classes.Add(new ClassItemViewModel(c));
        ActiveClass = Classes.FirstOrDefault(c => c.Id == (active ?? 2)) ?? Classes.FirstOrDefault();
    }

    // ───────────── Project lifecycle ─────────────

    [RelayCommand]
    private async Task NewProjectAsync()
    {
        if (!await ConfirmDiscardAsync()) return;
        var req = _dialogs.ShowNewProject(_settings.UserName);
        if (req is null) return;
        try
        {
            var p = _projects.Create(req);
            await _projects.SaveAsync(p, req.Folder);
            await LoadProjectAsync(p, req.Folder);
            _audit.Append(new AuditEntry(DateTimeOffset.UtcNow, $"user:{_settings.UserName}", "Project Created", null, null, p.Name));
            StatusMessage = "Project created. Import images to begin (toolbar → Import Images).";
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException or ArgumentException or UnauthorizedAccessException)
        {
            _dialogs.ShowError(ex.Message);
        }
    }

    [RelayCommand]
    private async Task OpenProjectAsync()
    {
        if (!await ConfirmDiscardAsync()) return;
        var folder = _dialogs.PickFolder("Select a CuAnnotator project folder");
        if (folder is not null) await OpenProjectFolderAsync(folder);
    }

    public async Task OpenProjectFolderAsync(string folder)
    {
        try
        {
            IsBusy = true;
            AnnotationProject p;
            if (_store.HasRecoveryData(folder) && _dialogs.Confirm("Autosaved changes newer than the last save were found (the application may have closed unexpectedly). Restore them?"))
                p = await _store.LoadRecoveryAsync(folder) ?? await _projects.OpenAsync(folder);
            else p = await _projects.OpenAsync(folder);
            if (_store.LastLoadWarning is { } w) _dialogs.ShowWarning(w);
            await LoadProjectAsync(p, folder);
            if (_store.HasRecoveryData(folder)) IsDirty = true;
            StatusMessage = $"Opened {p.Name}: {p.Images.Count} images, {p.Images.Sum(i => i.Annotations.Count)} annotations.";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or System.Text.Json.JsonException or UnauthorizedAccessException)
        {
            _log.Error(ex, "Failed to open project {Folder}", folder);
            _dialogs.ShowError($"Could not open the project:\n{ex.Message}");
        }
        finally { IsBusy = false; }
    }

    private async Task LoadProjectAsync(AnnotationProject p, string folder)
    {
        Editor?.Dispose();
        Editor = null;
        _histories.Clear();
        Project = p;
        ProjectFolder = folder;
        _audit = _store.OpenAuditLog(folder);
        LoadTaxonomy(p.Taxonomy);
        Parameters.Algorithm = p.Settings.DefaultAlgorithm;
        if (p.Settings.DefaultParameters.Count > 0)
        {
            var d = SegmentationParameters.FromDictionary(p.Settings.DefaultParameters);
            foreach (var prop in typeof(SegmentationParameters).GetProperties().Where(x => x.CanWrite)) prop.SetValue(Parameters, prop.GetValue(d));
        }
        OnPropertyChanged(nameof(Parameters));
        OnPropertyChanged(nameof(SelectedAlgorithm));
        Images.Clear();
        foreach (var i in p.Images) Images.Add(new ImageItemViewModel(i));
        IsDirty = false;
        _settings.AddRecent(folder);
        ConfigureAutosave();
        OnPropertyChanged(nameof(ProgressText));
        SelectedImage = Images.FirstOrDefault();
        await LoadThumbnailsAsync();
    }

    private void ConfigureAutosave()
    {
        _autosave.Stop();
        int s = Project?.Settings.AutosaveIntervalSeconds ?? 0;
        if (s > 0) { _autosave.Interval = TimeSpan.FromSeconds(Math.Max(15, s)); _autosave.Start(); }
        AutosaveStatus = s > 0 ? $"Autosave every {s}s" : "Autosave off";
    }

    private async Task AutosaveAsync()
    {
        if (Project is null || ProjectFolder is null || !IsDirty) return;
        try
        {
            await _store.SaveRecoveryAsync(Project, ProjectFolder);
            AutosaveStatus = $"Recovery data autosaved {DateTime.Now:HH:mm:ss}";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _log.Warning(ex, "Autosave failed");
            AutosaveStatus = "Autosave failed — see log";
        }
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private async Task SaveProjectAsync()
    {
        if (Project is null || ProjectFolder is null) return;
        try
        {
            IsBusy = true;
            if (Editor is not null) Project.Settings.DefaultParameters = Parameters.ToDictionary();
            await _projects.SaveAsync(Project, ProjectFolder);
            IsDirty = false;
            StatusMessage = $"Saved {DateTime.Now:HH:mm:ss} → {Path.Combine(ProjectFolder, NativeFormat.ProjectFileName)}";
            AutosaveStatus = "All changes saved";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException)
        {
            _log.Error(ex, "Save failed");
            _dialogs.ShowError($"Saving failed; the previous project file is unchanged.\n\n{ex.Message}");
        }
        finally { IsBusy = false; }
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private async Task SaveProjectAsAsync()
    {
        if (Project is null || ProjectFolder is null) return;
        var folder = _dialogs.PickFolder("Select an empty folder for the copy of this project");
        if (folder is null) return;
        if (File.Exists(Path.Combine(folder, NativeFormat.ProjectFileName)) && !_dialogs.Confirm("That folder already contains a project. Overwrite it?")) return;
        try
        {
            IsBusy = true;
            await _store.SaveAsAsync(Project, ProjectFolder, folder);
            ProjectFolder = folder;
            _audit = _store.OpenAuditLog(folder);
            IsDirty = false;
            _settings.AddRecent(folder);
            StatusMessage = $"Project saved as {folder}.";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { _dialogs.ShowError(ex.Message); }
        finally { IsBusy = false; }
    }

    /// <summary>Returns false if the user cancels.</summary>
    public async Task<bool> ConfirmDiscardAsync()
    {
        if (Project is null || !IsDirty) return true;
        var r = _dialogs.AskSaveChanges();
        if (r is null) return false;
        if (r == true) await SaveProjectAsync();
        else if (ProjectFolder is not null) _store.DiscardRecovery(ProjectFolder);
        return true;
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private async Task ImportImagesAsync()
    {
        if (Project is null || ProjectFolder is null) return;
        var files = _dialogs.OpenFiles("Images (*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp)|*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp");
        if (files.Length == 0) return;
        var meta = _dialogs.AskImportMetadata();
        if (meta is null) return;
        try
        {
            IsBusy = true;
            var progress = new Progress<(int Done, int Total, string Name)>(p => { Progress = p.Total == 0 ? 0 : 100.0 * p.Done / p.Total; StatusMessage = $"Importing {p.Name}…"; });
            var r = await _projects.ImportImagesAsync(Project, ProjectFolder, files, meta.Value.Specimen, meta.Value.Source, progress);
            foreach (var i in r.Imported)
            {
                i.Magnification = meta.Value.Magnification;
                i.MicrometersPerPixel = meta.Value.MicrometersPerPixel;
                Images.Add(new ImageItemViewModel(i));
                _audit.Append(new AuditEntry(DateTimeOffset.UtcNow, $"user:{_settings.UserName}", "Image Imported", i.Id, null,
                    $"{i.FileName} {i.Width}×{i.Height} {i.PixelFormat} sha256={i.Sha256}"));
            }
            MarkDirty();
            StatusMessage = $"Imported {r.Imported.Count} image(s)." + (r.Skipped.Count > 0 ? $" Skipped {r.Skipped.Count}." : "");
            if (r.Skipped.Count > 0) _dialogs.ShowWarning("Some files were not imported:\n\n" + string.Join("\n", r.Skipped));
            SelectedImage ??= Images.FirstOrDefault();
            if (r.Imported.Count > 0) SelectedImage = Images.First(i => i.Model == r.Imported[0]);
            await LoadThumbnailsAsync();
        }
        finally { IsBusy = false; Progress = 0; }
    }

    private async Task LoadThumbnailsAsync()
    {
        if (ProjectFolder is null) return;
        var folder = ProjectFolder;
        foreach (var vm in Images.Where(i => i.Thumbnail is null).ToList())
        {
            var path = ProjectPaths.Resolve(folder, vm.Model.RelativePath);
            var bmp = await Task.Run(() =>
            {
                try
                {
                    using var small = Cv2.ImRead(path, ImreadModes.ReducedColor4);
                    if (small.Empty()) return null;
                    using var t = new Mat();
                    double s = 120.0 / Math.Max(small.Width, small.Height);
                    Cv2.Resize(small, t, new OpenCvSharp.Size(Math.Max(1, (int)(small.Width * s)), Math.Max(1, (int)(small.Height * s))), 0, 0, InterpolationFlags.Area);
                    return TiledImageSource.ToBitmap(t);
                }
                catch (OpenCVException) { return null; }
            });
            vm.Thumbnail = bmp;
        }
    }

    // ───────────── Image navigation ─────────────

    partial void OnSelectedImageChanged(ImageItemViewModel? value) => _ = OpenImageAsync(value);

    private async Task OpenImageAsync(ImageItemViewModel? vm)
    {
        Editor?.Dispose();
        Editor = null;
        OnPropertyChanged(nameof(ImageIndexText));
        if (vm is null || Project is null || ProjectFolder is null) return;
        try
        {
            IsBusy = true;
            StatusMessage = $"Loading {vm.FileName}…";
            string? warning = null;
            var loaded = await Task.Run(() => ProjectService.LoadImage(ProjectFolder, vm.Model, out warning));
            if (SelectedImage != vm) { loaded.Dispose(); return; } // user moved on
            if (warning is not null) _dialogs.ShowWarning(warning);
            if (!_histories.TryGetValue(vm.Model.Id, out var history)) _histories[vm.Model.Id] = history = new UndoRedoStack();
            var session = new AnnotationSession(Project, vm.Model, loaded.Bgr, _engine, _audit, _settings.UserName, history: history);
            Editor = new EditorViewModel(this, session, loaded, _engine, _dialogs, _settings);
            ApplySceneOptions();
            RefreshUndoState();
            StatusMessage = $"{vm.FileName}: {vm.Model.Width} × {vm.Model.Height}, {vm.Model.PixelFormat}, {vm.Model.Annotations.Count} annotation(s). {ToolHints.For(ActiveTool)}";
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or OpenCVException or InvalidOperationException)
        {
            _log.Error(ex, "Failed to load image {File}", vm.FileName);
            StatusMessage = $"Cannot display {vm.FileName}: {ex.Message}";
            _dialogs.ShowError(StatusMessage);
        }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void NextImage() { if (Images.Count > 0) SelectedImage = Images[(Images.IndexOf(SelectedImage!) + 1 + Images.Count) % Images.Count]; }

    [RelayCommand]
    private void PreviousImage() { if (Images.Count > 0) SelectedImage = Images[(Images.IndexOf(SelectedImage!) - 1 + Images.Count) % Images.Count]; }

    [RelayCommand]
    private void SetWorkflowState(ImageWorkflowState state)
    {
        if (Editor is null) return;
        Editor.Session.SetWorkflowState(state);
        SelectedImage?.Refresh();
        MarkDirty();
    }

    [RelayCommand]
    private void EditImageMetadata()
    {
        if (SelectedImage is null) return;
        if (_dialogs.EditImageMetadata(SelectedImage.Model)) { SelectedImage.Refresh(); MarkDirty(); }
    }

    // ───────────── Editing commands ─────────────

    [RelayCommand(CanExecute = nameof(CanUndo))] private void Undo() { Editor?.Session.Undo(); MarkDirty(); }
    [RelayCommand(CanExecute = nameof(CanRedo))] private void Redo() { Editor?.Session.Redo(); MarkDirty(); }
    private bool CanUndo() => Editor?.Session.History.CanUndo ?? false;
    private bool CanRedo() => Editor?.Session.History.CanRedo ?? false;

    [RelayCommand] private void ZoomIn() => ViewCommandRequested?.Invoke(this, ViewCommand.ZoomIn);
    [RelayCommand] private void ZoomOut() => ViewCommandRequested?.Invoke(this, ViewCommand.ZoomOut);
    [RelayCommand] private void FitImage() => ViewCommandRequested?.Invoke(this, ViewCommand.Fit);
    [RelayCommand] private void ActualPixels() => ViewCommandRequested?.Invoke(this, ViewCommand.ActualPixels);
    [RelayCommand] private void ToggleOverlays() => OverlayMode = OverlayMode == OverlayMode.Hidden ? OverlayMode.Normal : OverlayMode.Hidden;
    [RelayCommand] private void SelectTool(ToolKind tool) => ActiveTool = tool;

    [RelayCommand]
    private void AssignClass(int classId)
    {
        ActiveClass = Classes.FirstOrDefault(c => c.Id == classId) ?? ActiveClass;
        Editor?.AssignClass(classId);
    }

    [RelayCommand]
    private void ToggleTheme()
    {
        _theme.Apply(_theme.Current == "Light" ? "Dark" : "Light");
        _settings.Theme = _theme.Current;
        _settings.Save();
        Editor?.Scene.Invalidate();
    }

    // ───────────── Dialog-based features ─────────────

    [RelayCommand(CanExecute = nameof(HasProject))]
    private async Task ExportDatasetAsync()
    {
        if (Project is null || ProjectFolder is null) return;
        var o = _dialogs.ShowExport(Project, Path.Combine(ProjectFolder, NativeFormat.ExportsFolder, $"dataset_{DateTime.Now:yyyyMMdd_HHmm}"));
        if (o is null) return;
        try
        {
            IsBusy = true;
            var progress = new Progress<(int Done, int Total, string Item)>(p => { Progress = 100.0 * p.Done / Math.Max(1, p.Total); StatusMessage = $"Exporting {p.Item}…"; });
            var report = await _exporter.ExportAsync(Project, ProjectFolder, o, progress);
            _audit.Append(new AuditEntry(DateTimeOffset.UtcNow, $"user:{_settings.UserName}", "Dataset Exported", null, null,
                $"{o.OutputFolder}; {report.FilesWritten.Count} files; {report.Warnings.Count} warnings"));
            StatusMessage = $"Exported {report.FilesWritten.Count} files to {o.OutputFolder}.";
            _dialogs.ShowText("Export report", $"Output: {o.OutputFolder}\nFiles written: {report.FilesWritten.Count}\n\n" +
                (report.Warnings.Count == 0 ? "No warnings. All exported masks were re-read and validated against the project.\n" : "Warnings:\n" + string.Join("\n", report.Warnings) + "\n") +
                (report.Messages.Count > 0 ? "\nNotes:\n" + string.Join("\n", report.Messages) : ""), null);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException or OpenCVException)
        {
            _log.Error(ex, "Export failed");
            _dialogs.ShowError($"Export failed: {ex.Message}");
        }
        finally { IsBusy = false; Progress = 0; }
    }

    [RelayCommand]
    private void ExportCurrentMask()
    {
        if (Editor is null) return;
        var path = _dialogs.SaveFile("PNG class-ID mask (*.png)|*.png", Path.GetFileNameWithoutExtension(Editor.Image.FileName) + "_mask.png");
        if (path is null) return;
        var map = LabelMapComposer.Compose(Editor.Image);
        MaskIO.WriteLabelPng(path, map.Labels, map.Width, map.Height);
        StatusMessage = $"Semantic mask ({map.Width}×{map.Height}, 8-bit class IDs) written to {path}.";
    }

    [RelayCommand]
    private void ExportSelectedBinaryMask()
    {
        if (Editor is null) return;
        var ids = Editor.SelectedIds.ToHashSet();
        var activeClass = ActiveClass?.Id;
        string what = ids.Count > 0 ? "selected annotation(s)" : $"class {ActiveClass?.Name}";
        Func<Annotation, bool> filter = ids.Count > 0 ? a => ids.Contains(a.Id) : a => a.ClassId == activeClass;
        var path = _dialogs.SaveFile("PNG binary mask (*.png)|*.png", Path.GetFileNameWithoutExtension(Editor.Image.FileName) + "_binary.png");
        if (path is null) return;
        MaskIO.WriteLabelPng(path, LabelMapComposer.Binary(Editor.Image, filter), Editor.Image.Width, Editor.Image.Height);
        StatusMessage = $"Binary mask (0/255) of {what} written to {path}.";
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private void TaxonomyEditor()
    {
        if (Project is null) return;
        var edited = _dialogs.ShowTaxonomyEditor(Project.Taxonomy, Project.UsedClassIds.ToList());
        if (edited is null) return;
        edited.Version = Project.Taxonomy.Version + 1;
        Project.Taxonomy = edited;
        LoadTaxonomy(edited);
        Editor?.SyncAnnotations();
        _audit.Append(new AuditEntry(DateTimeOffset.UtcNow, $"user:{_settings.UserName}", "Taxonomy Edited", null, null, $"version {edited.Version}, {edited.Classes.Count} classes"));
        MarkDirty();
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private void QualityReport()
    {
        if (Project is null) return;
        var images = SelectedImage is not null && _dialogs.Confirm("Analyse only the current image? (No = whole project)") ? new[] { SelectedImage.Model } : Project.Images.ToArray();
        var r = QualityAnalyzer.Analyze(Project, images, ProjectFolder);
        var sets = Project.Images.SelectMany(i => i.Annotations).Select(a => a.AnnotationSet).Distinct().ToList();
        var extra = "";
        if (sets.Count > 1 && SelectedImage is not null)
        {
            var (ag, bf) = EvaluationService.InterAnnotator(SelectedImage.Model, sets[0], sets[1]);
            extra = $"\nInter-annotator agreement on {SelectedImage.FileName} ('{sets[0]}' vs '{sets[1]}'):\n  pixel agreement {ag.PixelAgreement:P2}, Cohen's κ {ag.CohenKappa:0.000}, " +
                    $"mean IoU {ag.MeanIoU:0.000}, boundary F1 (2 px) {bf:0.000}\n";
        }
        var eff = EvaluationService.Summarize(Project);
        extra += $"\nAnnotation efficiency (recorded at acceptance):\n  {eff.Annotations} annotations; mean {eff.MeanSecondsPerRegion:0.0} s and {eff.MeanInteractions:0.0} interactions per region; " +
                 $"mean refinement runs {eff.MeanRefinementRuns:0.0}; first automatic proposal retained {eff.MeanFirstProposalRetained:P1}; " +
                 $"{eff.Segmented} segmentation-based, {eff.ManualOrEdited} edited/manual.\n";
        _dialogs.ShowText("Quality report", r.ToText() + extra, new Dictionary<string, Func<string>>
        {
            ["Class distribution (CSV)"] = r.ClassesCsv,
            ["Efficiency log (CSV)"] = () => EvaluationService.EfficiencyCsv(Project),
        });
        Editor?.Scene.Invalidate();
    }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private void ReviewQueue()
    {
        if (Project is null) return;
        var target = _dialogs.ShowReviewQueue(Project);
        if (target is null) return;
        SelectedImage = Images.FirstOrDefault(i => i.Model.Id == target.Value.ImageId);
        PendingSelection = target.Value.AnnotationId;
    }

    /// <summary>Annotation to select once the requested image has loaded (review queue navigation).</summary>
    public Guid? PendingSelection { get; set; }

    partial void OnEditorChanged(EditorViewModel? value)
    {
        if (value is not null && PendingSelection is { } id)
        {
            value.Select(id, false);
            var a = value.Image.Annotations.FirstOrDefault(x => x.Id == id);
            if (a is not null) ViewCenterRequested?.Invoke(this, new Domain.Geometry.ImagePoint(a.CentroidX, a.CentroidY));
            PendingSelection = null;
        }
        RefreshUndoState();
    }

    public event EventHandler<Domain.Geometry.ImagePoint>? ViewCenterRequested;

    [RelayCommand(CanExecute = nameof(HasProject))]
    private void AuditLog()
    {
        if (Project is null) return;
        var names = Project.Images.ToDictionary(i => i.Id, i => i.FileName);
        _dialogs.ShowText("Audit trail", JsonlAuditLog.Format(_audit.ReadAll(), id => names.GetValueOrDefault(id, id.ToString())), null);
    }

    [RelayCommand]
    private async Task EvaluationAsync()
    {
        if (Editor is null) { _dialogs.ShowInfo("Open an image first."); return; }
        var refPath = _dialogs.OpenFiles("Reference mask (*.png)|*.png").FirstOrDefault();
        if (refPath is null) return;
        var (labels, w, h) = MaskIO.ReadLabelPng(refPath);
        if (w != Editor.Image.Width || h != Editor.Image.Height) { _dialogs.ShowError($"Reference mask is {w}×{h}; the image is {Editor.Image.Width}×{Editor.Image.Height}."); return; }
        var roi = Editor.Session.Candidate?.Roi;
        if (roi is null) { _dialogs.ShowInfo("Draw an ROI first; algorithms are compared inside it."); return; }
        int cls = ActiveClass?.Id ?? 2;
        var bin = labels.Select(v => v == cls ? (byte)1 : (byte)0).ToArray(); // reference pixels of the active class
        var reference = new Domain.Geometry.RasterMask(new Domain.Geometry.PixelRect(0, 0, w, h), bin);
        IsBusy = true;
        try
        {
            var rows = await EvaluationService.CompareAlgorithmsAsync(_engine, Editor.Source, roi.Value, reference, Parameters, Algorithms);
            var text = $"Reference: {refPath} (class {cls}), ROI {roi}\n\n" + string.Join("\n", rows.Select(r =>
                $"{r.Algorithm,-34} IoU {r.Metrics.IoU:0.000}  Dice {r.Metrics.Dice:0.000}  P {r.Metrics.Precision:0.000}  R {r.Metrics.Recall:0.000}  BF1 {r.BoundaryF1:0.000}  {r.Milliseconds,6:0} ms"));
            _dialogs.ShowText("Algorithm evaluation", text + "\n\nBoundary F1 tolerance: 2 px. Results apply to this ROI only; do not generalise from a single region.",
                new Dictionary<string, Func<string>> { ["Results (CSV)"] = () => EvaluationService.ComparisonCsv(rows, 2.0) });
        }
        finally { IsBusy = false; }
    }

    [RelayCommand]
    private void CaptureBatchReferences()
    {
        if (Editor is null) return;
        BatchReferences = RegionSuggester.ClassReferences(Editor.Source, Editor.Image);
        BatchReferenceText = BatchReferences.Count == 0 ? "" :
            $"Batch references from {Editor.Image.FileName}: " + string.Join(", ", BatchReferences.Keys.Select(k => Project?.Taxonomy.Find(k)?.Name));
        StatusMessage = BatchReferences.Count == 0 ? "This image has no annotations to use as references."
            : "Class colour references captured. On other images use 'Find region suggestions'; each proposal still needs individual acceptance.";
    }

    [RelayCommand] private void ClearBatchReferences() { BatchReferences = null; BatchReferenceText = ""; }

    [RelayCommand(CanExecute = nameof(HasProject))]
    private void ProjectSettings()
    {
        if (Project is null) return;
        if (_dialogs.ShowProjectSettings(Project, _settings))
        {
            ConfigureAutosave();
            OnPropertyChanged(nameof(SeverityLevels));
            MarkDirty();
        }
    }

    [RelayCommand] private void Help() => _dialogs.ShowHelp();
    [RelayCommand] private void Onboarding() => _dialogs.ShowOnboarding();

    public UserSettings Settings => _settings;
    public IReadOnlyList<string> RecentProjects => _settings.RecentProjects;

    [RelayCommand]
    private async Task OpenRecentAsync(string? folder)
    {
        if (folder is null || !await ConfirmDiscardAsync()) return;
        await OpenProjectFolderAsync(folder);
    }
}

/// <summary>Context-sensitive one-line help for each tool (status bar).</summary>
public static class ToolHints
{
    public static string For(ToolKind t) => t switch
    {
        ToolKind.Select => "Select / Pan: click an annotation to select (Ctrl+click adds); drag to pan.",
        ToolKind.Rectangle => "Rectangle ROI: drag loosely around a corrosion region; segmentation runs automatically.",
        ToolKind.Polygon => "Polygon ROI: click vertices, double-click or Enter to close; the polygon seeds the segmentation.",
        ToolKind.Lasso => "Freehand lasso: drag around the region; it becomes the initial mask.",
        ToolKind.SmartContour => "Smart contour: drag a rectangle; runs the hybrid GrabCut + CIELAB + edge pipeline.",
        ToolKind.ForegroundBrush => "Foreground brush: paint pixels that must belong to the region (candidate), or add to the selected annotation.",
        ToolKind.BackgroundBrush => "Background brush: paint pixels that must NOT belong to the region.",
        ToolKind.BoundaryBrush => "Boundary brush: trace roughly along an edge; the path snaps to image gradients and acts as a barrier.",
        ToolKind.Eraser => "Eraser: remove pixels from the selected annotation.",
        ToolKind.VertexEditor => "Vertex editor: drag vertices; double-click an edge to add; right-click a vertex to delete; Ctrl+drag moves the region.",
        ToolKind.Split => "Split: select an annotation, then draw a cut line across it.",
        ToolKind.Merge => "Merge: Ctrl+click annotations to select several, then press Enter or the Merge button.",
        ToolKind.Fill => "Fill: click an annotation to fill its interior holes.",
        ToolKind.ColorSelect => "Colour selection: click a sample pixel; similar pixels (ΔE ≤ threshold) in the ROI or image are selected.",
        ToolKind.MagicWand => "Magic wand: click to grow a connected region of similar CIELAB colour.",
        ToolKind.Measure => "Measure / inspect: hover to inspect pixel values; drag to measure a distance.",
        _ => "",
    };
}
