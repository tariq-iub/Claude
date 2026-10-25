namespace CuAnnotator.Domain.Enums;

/// <summary>
/// How well the chemical-phase identity of an annotation is supported. Ordered from weakest to strongest.
/// An algorithmic class suggestion never changes this value.
/// </summary>
public enum VerificationStatus
{
    Unverified = 0,
    VisuallyEstimated = 1,
    ExpertReviewed = 2,
    InstrumentallyVerified = 3,
}

public enum EvidenceType
{
    XRD,
    Raman,
    FTIR,
    SemEds,
    XPS,
    LaboratoryNotes,
    PublishedReference,
    ExpertObservation,
    Other,
}

/// <summary>Review workflow state of an image.</summary>
public enum ImageWorkflowState
{
    Unannotated,
    InProgress,
    Annotated,
    PendingReview,
    Verified,
}

/// <summary>What to do when a newly accepted region overlaps already accepted annotations.</summary>
public enum OverlapPolicy
{
    /// <summary>The new region loses: overlapping pixels are removed from the new annotation.</summary>
    PreserveExisting,
    /// <summary>The new region wins: overlapping pixels are removed from the existing annotations.</summary>
    ReplaceExisting,
    /// <summary>Keep overlap; semantic export uses the topmost (highest Z) annotation and QA reports it.</summary>
    AllowOverlapTopmostWins,
    /// <summary>Ask the user each time (UI resolves to one of the other policies).</summary>
    AskUser,
}

/// <summary>Origin of the current geometry of an annotation.</summary>
public enum GeometrySource
{
    Segmentation,
    PolygonEdit,
    BrushEdit,
    Merge,
    Split,
    BooleanOperation,
    Import,
}

/// <summary>How the class label of an annotation was chosen.</summary>
public enum ClassAssignmentSource
{
    Manual,
    AcceptedSuggestion,
    BatchDefault,
}

public enum SegmentationAlgorithmKind
{
    GrabCut,
    LabColor,
    Watershed,
    ActiveContour,
    Hybrid,
    MagicWand,
    Manual,
}
