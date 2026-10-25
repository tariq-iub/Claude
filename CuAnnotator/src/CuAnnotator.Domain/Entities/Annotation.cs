using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Domain.Entities;

public sealed class EvidenceReference
{
    public EvidenceType Type { get; set; }
    /// <summary>Instrument run ID, file name, DOI, notebook page …</summary>
    public string Source { get; set; } = "";
    public string Notes { get; set; } = "";
    public DateTimeOffset? Date { get; set; }
    public EvidenceReference Clone() => (EvidenceReference)MemberwiseClone();
}

/// <summary>
/// A class suggestion produced by the class-assisted subsystem. Kept separate from the
/// annotation's class and verification status: it is a visual prediction, not a label.
/// </summary>
public sealed class ClassSuggestion
{
    public int ClassId { get; set; }
    /// <summary>Uncalibrated similarity in [0,1]; not a probability.</summary>
    public double SimilarityScore { get; set; }
    public string Basis { get; set; } = "";
    public DateTimeOffset CreatedAt { get; set; }
    public ClassSuggestion Clone() => (ClassSuggestion)MemberwiseClone();
}

/// <summary>Immutable snapshot of an annotation at a revision.</summary>
public sealed class AnnotationRevision
{
    public int Revision { get; set; }
    public DateTimeOffset Timestamp { get; set; }
    public string Author { get; set; } = "";
    public string Reason { get; set; } = "";
    public int ClassId { get; set; }
    public VerificationStatus VerificationStatus { get; set; }
    public GeometrySource GeometrySource { get; set; }
    public RasterMask Mask { get; set; } = RasterMask.Empty;
    public string Notes { get; set; } = "";
}

/// <summary>
/// One annotated region (instance). The <see cref="Mask"/> is the authoritative geometry;
/// <see cref="Polygons"/> is the editable vector view derived from it (or, after a polygon edit,
/// the source from which the mask was re-rasterised by the documented rule).
/// </summary>
public sealed class Annotation
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid ImageId { get; set; }
    public int ClassId { get; set; }
    public string DisplayName { get; set; } = "";
    public ClassAssignmentSource ClassAssignmentSource { get; set; } = ClassAssignmentSource.Manual;

    public RasterMask Mask { get; set; } = RasterMask.Empty;
    public List<PolygonComponent> Polygons { get; set; } = new();
    public GeometrySource GeometrySource { get; set; } = GeometrySource.Segmentation;

    // Derived measurements (refreshed by AnnotationGeometryService whenever the mask changes).
    public PixelRect BoundingBox { get; set; }
    public long PixelArea { get; set; }
    public double CentroidX { get; set; }
    public double CentroidY { get; set; }
    /// <summary>Perimeter of the polygon outlines in pixels (sum over components and holes).</summary>
    public double Perimeter { get; set; }

    public string SegmentationAlgorithm { get; set; } = "";
    public string AlgorithmVersion { get; set; } = "";
    public Dictionary<string, string> AlgorithmParameters { get; set; } = new();

    public VerificationStatus VerificationStatus { get; set; } = VerificationStatus.Unverified;
    public List<EvidenceReference> Evidence { get; set; } = new();
    public string Notes { get; set; } = "";
    public string? Severity { get; set; }
    /// <summary>Reviewer assessment of the annotation, 0–1, set by a person (not an algorithm).</summary>
    public double? ReviewerConfidence { get; set; }
    public ClassSuggestion? Suggestion { get; set; }
    /// <summary>Feature descriptor used by class-assisted suggestions.</summary>
    public double[]? Descriptor { get; set; }

    public DateTimeOffset CreatedAt { get; set; }
    public DateTimeOffset ModifiedAt { get; set; }
    public string CreatedBy { get; set; } = "";
    public string? ReviewedBy { get; set; }
    public DateTimeOffset? ReviewedAt { get; set; }
    public int Revision { get; set; } = 1;
    public List<AnnotationRevision> RevisionHistory { get; set; } = new();
    public bool IsVisible { get; set; } = true;
    /// <summary>Stacking order (higher draws on top and wins under AllowOverlapTopmostWins).</summary>
    public int ZOrder { get; set; }
    /// <summary>Annotator set name used for inter-annotator comparison ("default" for the main set).</summary>
    public string AnnotationSet { get; set; } = "default";

    public Annotation DeepClone() => new()
    {
        Id = Id, ImageId = ImageId, ClassId = ClassId, DisplayName = DisplayName,
        ClassAssignmentSource = ClassAssignmentSource,
        Mask = Mask, // masks are immutable by convention
        Polygons = Polygons.Select(p => p.Clone()).ToList(),
        GeometrySource = GeometrySource, BoundingBox = BoundingBox, PixelArea = PixelArea,
        CentroidX = CentroidX, CentroidY = CentroidY, Perimeter = Perimeter,
        SegmentationAlgorithm = SegmentationAlgorithm, AlgorithmVersion = AlgorithmVersion,
        AlgorithmParameters = new(AlgorithmParameters),
        VerificationStatus = VerificationStatus, Evidence = Evidence.Select(e => e.Clone()).ToList(),
        Notes = Notes, Severity = Severity, ReviewerConfidence = ReviewerConfidence,
        Suggestion = Suggestion?.Clone(), Descriptor = (double[]?)Descriptor?.Clone(),
        CreatedAt = CreatedAt, ModifiedAt = ModifiedAt, CreatedBy = CreatedBy,
        ReviewedBy = ReviewedBy, ReviewedAt = ReviewedAt, Revision = Revision,
        RevisionHistory = RevisionHistory.ToList(), IsVisible = IsVisible, ZOrder = ZOrder,
        AnnotationSet = AnnotationSet,
    };

    /// <summary>Copies every field of <paramref name="source"/> into this instance (identity preserved).</summary>
    public void RestoreFrom(Annotation source)
    {
        var s = source.DeepClone();
        foreach (var prop in typeof(Annotation).GetProperties().Where(p => p.CanWrite))
            prop.SetValue(this, prop.GetValue(s));
    }
}
