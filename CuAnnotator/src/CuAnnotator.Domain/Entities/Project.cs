using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Domain.Entities;

public sealed class ImageItem
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string FileName { get; set; } = "";
    /// <summary>Path relative to the project folder (preferred) or absolute when outside it.</summary>
    public string RelativePath { get; set; } = "";
    public string? OriginalSourcePath { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    /// <summary>Pixel format as decoded (e.g. "8UC3", "16UC3", "8UC1").</summary>
    public string PixelFormat { get; set; } = "";
    public int BitDepth { get; set; } = 8;
    public int Channels { get; set; } = 3;
    public long FileSizeBytes { get; set; }
    public string Sha256 { get; set; } = "";
    public DateTimeOffset ImportedAt { get; set; }
    public string? ColorProfile { get; set; }
    /// <summary>EXIF orientation tag (1–8) as found in the file; 1 or null means none. Pixels are stored upright.</summary>
    public int? ExifOrientation { get; set; }
    public bool ExifOrientationApplied { get; set; }
    public string? SpecimenId { get; set; }
    public string? AcquisitionSource { get; set; }
    public double? Magnification { get; set; }
    public double? MicrometersPerPixel { get; set; }
    public string Notes { get; set; } = "";
    public ImageWorkflowState WorkflowState { get; set; } = ImageWorkflowState.Unannotated;
    public List<Annotation> Annotations { get; set; } = new();

    /// <summary>Specimen grouping key used for leakage-free dataset splitting.</summary>
    public string SpecimenKey => string.IsNullOrWhiteSpace(SpecimenId) ? $"image:{Id}" : SpecimenId!;
}

public sealed class ExportPreferences
{
    public bool IncludeColorizedMasks { get; set; } = true;
    public bool IncludeCoco { get; set; } = true;
    public bool IncludeYolo { get; set; }
    public bool IncludeNativeJson { get; set; } = true;
    public bool CopyImagesAsPng { get; set; } = true;
    public double TrainFraction { get; set; } = 0.7;
    public double ValidationFraction { get; set; } = 0.15;
    public int SplitSeed { get; set; } = 42;
    public bool GroupBySpecimen { get; set; } = true;
}

public sealed class ProjectSettings
{
    public SegmentationAlgorithmKind DefaultAlgorithm { get; set; } = SegmentationAlgorithmKind.GrabCut;
    public Dictionary<string, string> DefaultParameters { get; set; } = new();
    public OverlapPolicy OverlapPolicy { get; set; } = OverlapPolicy.PreserveExisting;
    public int AutosaveIntervalSeconds { get; set; } = 120;
    public bool CopyImagesIntoProject { get; set; } = true;
    public List<string> SeverityLevels { get; set; } = new() { "Mild", "Moderate", "Severe" };
    public string SeverityCriteria { get; set; } = "";
    public ExportPreferences Export { get; set; } = new();
}

public sealed class AnnotationProject
{
    /// <summary>Native schema version written to every project file.</summary>
    public const int CurrentSchemaVersion = 1;

    public int SchemaVersion { get; set; } = CurrentSchemaVersion;
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public string Researcher { get; set; } = "";
    public string Institution { get; set; } = "";
    public DateTimeOffset CreatedAt { get; set; }
    public DateTimeOffset ModifiedAt { get; set; }
    public string CreatedWithAppVersion { get; set; } = "";
    public string SavedWithAppVersion { get; set; } = "";
    public Taxonomy Taxonomy { get; set; } = Taxonomy.CreateDefault();
    public ProjectSettings Settings { get; set; } = new();
    public List<ImageItem> Images { get; set; } = new();

    public ImageItem? FindImage(Guid id) => Images.FirstOrDefault(i => i.Id == id);

    public IEnumerable<int> UsedClassIds => Images.SelectMany(i => i.Annotations).Select(a => a.ClassId).Distinct();
}

/// <summary>An audit-trail entry. <see cref="Actor"/> distinguishes user actions from algorithm output.</summary>
public sealed record AuditEntry(
    DateTimeOffset Timestamp,
    string Actor,          // "user:<name>" or "algorithm:<name>@<version>" or "system"
    string Action,
    Guid? ImageId,
    Guid? AnnotationId,
    string Details);

public interface IAuditLog
{
    void Append(AuditEntry entry);
    IReadOnlyList<AuditEntry> ReadAll();
}
