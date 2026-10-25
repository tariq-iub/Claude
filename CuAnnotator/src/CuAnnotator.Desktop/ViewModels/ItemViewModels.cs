using System.Globalization;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Desktop.Rendering;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Desktop.ViewModels;

public sealed class ClassItemViewModel(CorrosionClass model) : ObservableObject
{
    public CorrosionClass Model { get; } = model;
    public int Id => Model.Id;
    public string Name => Model.Name;
    public string Display => string.IsNullOrEmpty(Model.ChemicalFormula) || Model.ChemicalFormula == Model.Name ? Model.Name : $"{Model.Name}  ({Model.ChemicalFormula})";
    public string? Shortcut => Model.Shortcut;
    public string Description => Model.Description;
    public string? Parent => Model.ParentCategory;
    public bool IsEnabled => Model.IsEnabled;
    public Brush ColorBrush => Frozen(new SolidColorBrush(OverlayFactory.ParseColor(Model.ColorHex)));
    public void Refresh() => OnPropertyChanged(string.Empty);

    internal static Brush Frozen(Brush b) { b.Freeze(); return b; }
}

public sealed class ImageItemViewModel(ImageItem model) : ObservableObject
{
    public ImageItem Model { get; } = model;
    public string FileName => Model.FileName;
    public string Dimensions => $"{Model.Width} × {Model.Height}";
    public int AnnotationCount => Model.Annotations.Count;
    public ImageWorkflowState State => Model.WorkflowState;
    public string Specimen => Model.SpecimenId ?? "";

    private ImageSource? _thumbnail;
    public ImageSource? Thumbnail { get => _thumbnail; set => SetProperty(ref _thumbnail, value); }

    public void Refresh() => OnPropertyChanged(string.Empty);
}

/// <summary>
/// Properties-panel view of an annotation. Every edit goes through <see cref="AnnotationSession"/> so it is
/// undoable and recorded in the audit trail.
/// </summary>
public sealed class AnnotationViewModel(Annotation model, AnnotationSession session, Taxonomy taxonomy) : ObservableObject
{
    public Annotation Model { get; } = model;
    public Guid Id => Model.Id;
    public string ShortId => Model.Id.ToString()[..8];
    public string DisplayName
    {
        get => Model.DisplayName;
        set { if (value != Model.DisplayName) session.UpdateMetadata(Model.Id, "Rename annotation", a => a.DisplayName = value); }
    }
    public int ClassId => Model.ClassId;
    public string ClassName => taxonomy.Find(Model.ClassId)?.Name ?? $"? ({Model.ClassId})";
    public Brush ClassBrush => ClassItemViewModel.Frozen(new SolidColorBrush(OverlayFactory.ParseColor(taxonomy.Find(Model.ClassId)?.ColorHex ?? "#FF00FF")));
    public long PixelArea => Model.PixelArea;
    public string AreaText => Model.PixelArea.ToString("N0") + " px";
    public string BoundingBoxText => $"x {Model.BoundingBox.X}, y {Model.BoundingBox.Y}, {Model.BoundingBox.Width} × {Model.BoundingBox.Height}";
    public string CentroidText => $"({Model.CentroidX:0.0}, {Model.CentroidY:0.0})";
    public string PerimeterText => $"{Model.Perimeter:0.0} px";
    public int VertexCount => Model.Polygons.Sum(p => p.VertexCount);
    public string ComponentsText => $"{Model.Polygons.Count} component(s), {Model.Polygons.Sum(p => p.Holes.Count)} hole(s), {VertexCount} vertices";
    public string Algorithm => string.IsNullOrEmpty(Model.SegmentationAlgorithm) ? "—" : $"{Model.SegmentationAlgorithm} v{Model.AlgorithmVersion}";
    public string GeometrySourceText => Model.GeometrySource.ToString();
    public string ParametersText => string.Join("\n", Model.AlgorithmParameters.OrderBy(k => k.Key).Select(kv => $"{kv.Key} = {kv.Value}"));
    public string CreatedText => $"{Model.CreatedAt.ToLocalTime():yyyy-MM-dd HH:mm:ss} by {Model.CreatedBy}";
    public string ModifiedText => Model.ModifiedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.CurrentCulture);
    public string ReviewText => Model.ReviewedBy is null ? "Not reviewed" : $"{Model.ReviewedBy} ({Model.ReviewedAt?.ToLocalTime():yyyy-MM-dd HH:mm})";
    public int Revision => Model.Revision;
    public IReadOnlyList<AnnotationRevision> Revisions => Model.RevisionHistory.OrderByDescending(r => r.Revision).ToList();
    public IReadOnlyList<EvidenceReference> Evidence => Model.Evidence.ToList();
    public string AssignmentText => Model.ClassAssignmentSource switch
    {
        ClassAssignmentSource.AcceptedSuggestion => "Class from accepted suggestion (visual prediction)",
        ClassAssignmentSource.BatchDefault => "Class from batch assistance",
        _ => "Class assigned manually",
    };
    public string SuggestionText => Model.Suggestion is { } s
        ? $"Suggested: {taxonomy.Find(s.ClassId)?.Name ?? s.ClassId.ToString()} — similarity {s.SimilarityScore:0.00} (uncalibrated, not a probability)\n{s.Basis}"
        : "";
    public bool HasSuggestion => Model.Suggestion is not null;

    public VerificationStatus Verification
    {
        get => Model.VerificationStatus;
        set { if (value != Model.VerificationStatus) session.SetVerification(Model.Id, value); }
    }

    public string Notes
    {
        get => Model.Notes;
        set { if (value != Model.Notes) session.UpdateMetadata(Model.Id, "Edit notes", a => a.Notes = value); }
    }

    public string? Severity
    {
        get => Model.Severity;
        set { if (value != Model.Severity) session.UpdateMetadata(Model.Id, "Set severity", a => a.Severity = string.IsNullOrEmpty(value) ? null : value); }
    }

    public double? ReviewerConfidence
    {
        get => Model.ReviewerConfidence;
        set { if (value != Model.ReviewerConfidence) session.UpdateMetadata(Model.Id, "Set reviewer assessment", a => a.ReviewerConfidence = value is null ? null : Math.Clamp(value.Value, 0, 1)); }
    }

    public bool IsVisible
    {
        get => Model.IsVisible;
        set { if (value != Model.IsVisible) session.SetVisibility(Model.Id, value); }
    }

    public void Refresh() => OnPropertyChanged(string.Empty);
}
