using System.Globalization;
using CuAnnotator.Application.Commands;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Application.Annotations;

using Annotation = Domain.Entities.Annotation;

public enum GuidanceKind { Foreground, Background, Barrier }

/// <summary>
/// The in-progress (not yet accepted) segmentation. Immutable: every change creates a new instance so
/// undo/redo can swap candidates atomically.
/// </summary>
public sealed record CandidateState
{
    public required PixelRect Roi { get; init; }
    public required SegmentationParameters Parameters { get; init; }
    public Guid? EditingAnnotationId { get; init; }
    public RasterMask? InitialMask { get; init; }
    public IReadOnlyList<Stroke> Positive { get; init; } = Array.Empty<Stroke>();
    public IReadOnlyList<Stroke> Negative { get; init; } = Array.Empty<Stroke>();
    public IReadOnlyList<Stroke> Barrier { get; init; } = Array.Empty<Stroke>();
    public SegmentationResult? Result { get; init; }
    /// <summary>First automatic proposal (for the "automatically proposed pixels retained" metric).</summary>
    public RasterMask? FirstProposal { get; init; }
    public int Interactions { get; init; }
    public int Runs { get; init; }
    public DateTimeOffset StartedAt { get; init; }
}

public sealed record AcceptResult(Annotation Annotation, long OverlapPixels, IReadOnlyList<Guid> ModifiedAnnotations, IReadOnlyList<Guid> RemovedAnnotations);

/// <summary>
/// UI-independent editing engine for one image: ROI → segmentation → guidance → refinement →
/// acceptance → class/verification → geometry edits, with undo/redo and an audit trail.
/// All mutating members must be called from one thread (the UI thread in the desktop app);
/// segmentation itself runs on the thread pool.
/// </summary>
public sealed class AnnotationSession
{
    private readonly SegmentationEngine _engine;
    private readonly IAuditLog _audit;
    private readonly TimeProvider _clock;
    private long _candidateVersion;

    public AnnotationProject Project { get; }
    public ImageItem Image { get; }
    /// <summary>Full-resolution BGR source (owned by the caller).</summary>
    public Mat Source { get; }
    public string User { get; set; }
    public UndoRedoStack History { get; }
    public CandidateState? Candidate { get; private set; }
    public bool IsSegmenting { get; private set; }

    public event EventHandler? CandidateChanged;
    public event EventHandler? AnnotationsChanged;

    public int Width => Source.Width;
    public int Height => Source.Height;

    public AnnotationSession(AnnotationProject project, ImageItem image, Mat source, SegmentationEngine engine,
        IAuditLog audit, string user, TimeProvider? clock = null, UndoRedoStack? history = null)
    {
        History = history ?? new UndoRedoStack();
        if (source.Width != image.Width || source.Height != image.Height)
            throw new InvalidOperationException($"Decoded image is {source.Width}×{source.Height} but the project records {image.Width}×{image.Height}.");
        Project = project; Image = image; Source = source; _engine = engine; _audit = audit; User = user;
        _clock = clock ?? TimeProvider.System;
    }

    private DateTimeOffset Now => _clock.GetUtcNow();
    private string UserActor => $"user:{User}";

    private void Log(string actor, string action, Guid? annotationId, string details) =>
        _audit.Append(new AuditEntry(Now, actor, action, Image.Id, annotationId, details));

    // ───────────────────────── Candidate lifecycle ─────────────────────────

    private void SetCandidate(CandidateState? next, string description, bool undoable = true)
    {
        var before = Candidate;
        void Apply(CandidateState? s) { Candidate = s; _candidateVersion++; CandidateChanged?.Invoke(this, EventArgs.Empty); }
        if (undoable) History.Execute(new DelegateUndoableCommand(description, () => Apply(next), () => Apply(before)));
        else Apply(next);
    }

    public void BeginRoi(PixelRect roi, SegmentationParameters parameters)
    {
        var clipped = roi.ClampTo(Width, Height);
        if (clipped.Width < 3 || clipped.Height < 3) throw new ArgumentException("ROI must cover at least 3×3 image pixels.");
        SetCandidate(new CandidateState { Roi = clipped, Parameters = parameters.Clone(), StartedAt = Now, Interactions = 1 }, "Create ROI");
        Log(UserActor, "ROI Created", null, $"Coordinates: ({clipped.X}, {clipped.Y}, {clipped.Right}, {clipped.Bottom})");
    }

    /// <summary>
    /// Starts from a freehand lasso or polygon ROI: the ROI is its bounding box and the drawn region becomes the
    /// initial mask (GrabCut: inside = probable foreground, outside = probable background).
    /// </summary>
    public void BeginRegion(RasterMask region, SegmentationParameters parameters, string kind)
    {
        var r = region.ClipToImage(Width, Height);
        if (r.IsEmpty) throw new ArgumentException("The drawn region does not cover any image pixels.");
        var roi = r.Bounds;
        if (roi.Width < 3 || roi.Height < 3) throw new ArgumentException("The drawn region is too small (minimum 3×3 pixels).");
        SetCandidate(new CandidateState { Roi = roi, Parameters = parameters.Clone(), InitialMask = r, StartedAt = Now, Interactions = 1 }, $"Create {kind} ROI");
        Log(UserActor, $"{kind} ROI Created", null, $"Bounds: ({roi.X}, {roi.Y}, {roi.Right}, {roi.Bottom}); {r.PixelCount} px");
    }

    /// <summary>Shows an externally computed result (magic wand, colour selection, region suggestion) as the candidate.</summary>
    public void ProposeCandidate(SegmentationResult result, string description)
    {
        if (result.Mask.IsEmpty) throw new InvalidOperationException("The proposal is empty.");
        var roi = result.Mask.Bounds.Inflate(2).ClampTo(Width, Height);
        SetCandidate(new CandidateState
        {
            Roi = roi, Parameters = SegmentationParameters.FromDictionary(result.ParametersUsed), StartedAt = Now, Interactions = 1,
            Result = result, FirstProposal = result.Mask, Runs = 1,
        }, description);
        Log($"algorithm:{result.AlgorithmName}@{result.AlgorithmVersion}", description, null,
            $"Candidate boundary: {result.Mask.PixelCount} px, {result.Components.Count} component(s)");
    }

    /// <summary>Starts re-segmentation of an accepted annotation; its mask becomes the initial mask.</summary>
    public void BeginEditExisting(Guid annotationId, SegmentationParameters parameters, int margin = 20)
    {
        var a = Find(annotationId);
        var roi = a.Mask.ContentBounds().Inflate(margin).ClampTo(Width, Height);
        SetCandidate(new CandidateState
        {
            Roi = roi, Parameters = parameters.Clone(), EditingAnnotationId = a.Id, InitialMask = a.Mask, StartedAt = Now, Interactions = 1,
        }, "Re-segment annotation");
        Log(UserActor, "Re-segmentation Started", a.Id, $"ROI {roi}");
    }

    public void AddGuidance(GuidanceKind kind, Stroke stroke)
    {
        var c = Candidate ?? throw new InvalidOperationException("Draw an ROI before adding guidance.");
        var next = kind switch
        {
            GuidanceKind.Foreground => c with { Positive = c.Positive.Append(stroke).ToList() },
            GuidanceKind.Background => c with { Negative = c.Negative.Append(stroke).ToList() },
            _ => c with { Barrier = c.Barrier.Append(stroke).ToList() },
        };
        SetCandidate(next with { Interactions = c.Interactions + 1 }, $"{kind} guidance");
        Log(UserActor, $"{kind} Correction", c.EditingAnnotationId, $"{stroke.Points.Count} points, radius {stroke.Radius}");
    }

    public void UpdateParameters(SegmentationParameters parameters)
    {
        if (Candidate is null) return;
        SetCandidate(Candidate with { Parameters = parameters.Clone() }, "Change parameters", undoable: false);
    }

    public void ClearGuidance()
    {
        if (Candidate is null) return;
        SetCandidate(Candidate with { Positive = Array.Empty<Stroke>(), Negative = Array.Empty<Stroke>(), Barrier = Array.Empty<Stroke>() }, "Clear guidance");
    }

    public void CancelCandidate()
    {
        if (Candidate is null) return;
        SetCandidate(null, "Cancel segmentation");
        Log(UserActor, "Segmentation Cancelled", null, "");
    }

    /// <summary>
    /// Runs the selected algorithm on the current candidate in the background. If the candidate changes
    /// while the algorithm runs (new stroke, undo, cancel), the stale result is discarded so history is
    /// never corrupted.
    /// </summary>
    public async Task<SegmentationResult?> RunSegmentationAsync(CancellationToken ct = default)
    {
        var c = Candidate ?? throw new InvalidOperationException("No ROI selected.");
        long version = _candidateVersion;
        var request = new SegmentationRequest
        {
            Image = Source, Roi = c.Roi, Parameters = c.Parameters,
            // Re-runs continue from the previous result; GrabCut instead reuses its own 4-state mask and GMMs.
            InitialMask = c.Result is null ? c.InitialMask
                : c.Parameters.Algorithm is SegmentationAlgorithmKind.GrabCut or SegmentationAlgorithmKind.Hybrid ? null : c.Result.Mask,
            PositiveStrokes = c.Positive, NegativeStrokes = c.Negative, BarrierStrokes = c.Barrier,
            PreviousGrabCutState = c.Result?.GrabCutState,
        };
        IsSegmenting = true;
        CandidateChanged?.Invoke(this, EventArgs.Empty);
        try
        {
            var result = await _engine.SegmentAsync(request, ct);
            if (version != _candidateVersion || Candidate is null) return null; // stale
            var algoActor = $"algorithm:{result.AlgorithmName}@{result.AlgorithmVersion}";
            Log(algoActor, $"{result.AlgorithmName} Segmentation", c.EditingAnnotationId,
                $"Candidate boundary: {result.Mask.PixelCount} px, {result.Components.Count} component(s), {result.ProcessingTime.TotalMilliseconds:0} ms; " +
                string.Join(", ", KeyParams(result.ParametersUsed)));
            SetCandidate(c with { Result = result, FirstProposal = c.FirstProposal ?? result.Mask, Runs = c.Runs + 1 }, "Run segmentation");
            return result;
        }
        finally
        {
            IsSegmenting = false;
            CandidateChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    private static IEnumerable<string> KeyParams(Dictionary<string, string> p)
    {
        foreach (var k in new[] { "Algorithm", "GrabCutIterations", "ColorThreshold", "ColorMetric", "MinComponentArea", "Quality" })
            if (p.TryGetValue(k, out var v)) yield return $"{k}={v}";
    }

    /// <summary>Overlap (pixels) between a mask and accepted annotations of the same annotation set.</summary>
    public long OverlapWithExisting(RasterMask mask, Guid? exclude = null, string set = "default") =>
        Image.Annotations.Where(a => a.Id != exclude && a.AnnotationSet == set).Sum(a => a.Mask.OverlapCount(mask));

    /// <summary>Accepts the current candidate as a new annotation (or as a new revision of the edited one).</summary>
    public AcceptResult AcceptCandidate(int classId, VerificationStatus verification, string notes = "",
        OverlapPolicy? policy = null, string annotationSet = "default")
    {
        var c = Candidate ?? throw new InvalidOperationException("Nothing to accept.");
        var result = c.Result ?? throw new InvalidOperationException("Run segmentation before accepting.");
        if (result.Mask.IsEmpty) throw new InvalidOperationException("The candidate segmentation is empty.");
        var cls = Project.Taxonomy.Find(classId) ?? throw new ArgumentException($"Unknown class {classId}.");
        if (classId == Taxonomy.UnlabeledId) throw new ArgumentException("Class 0 (Unlabeled) cannot be assigned.");
        var pol = policy ?? Project.Settings.OverlapPolicy;
        if (pol == OverlapPolicy.AskUser) throw new InvalidOperationException("Overlap policy 'AskUser' must be resolved by the UI before accepting.");

        Annotation? created = null;
        var modified = new List<Guid>(); var removed = new List<Guid>();
        long overlap = 0;
        var editing = c.EditingAnnotationId is { } eid ? Find(eid) : null;

        var cmd = new ImageSnapshotCommand(Image, editing is null ? "Accept segmentation" : "Accept re-segmentation", () =>
        {
            var mask = result.Mask;
            overlap = OverlapWithExisting(mask, editing?.Id, annotationSet);
            var others = Image.Annotations.Where(a => a.Id != editing?.Id && a.AnnotationSet == annotationSet).ToList();
            if (overlap > 0 && pol == OverlapPolicy.PreserveExisting)
                foreach (var o in others) mask = mask.Subtract(o.Mask);
            if (mask.IsEmpty) throw new InvalidOperationException("The region lies entirely inside existing annotations (policy: preserve existing).");

            var target = editing ?? new Annotation
            {
                ImageId = Image.Id, CreatedAt = Now, CreatedBy = User, Revision = 0, AnnotationSet = annotationSet,
                ZOrder = Image.Annotations.Count == 0 ? 0 : Image.Annotations.Max(a => a.ZOrder) + 1,
            };
            if (editing is not null) PushRevision(target, "Re-segmentation");
            AnnotationGeometryService.SetMask(target, mask);
            target.Revision++;
            target.ClassId = classId;
            target.DisplayName = string.IsNullOrEmpty(target.DisplayName) ? $"{cls.Name} #{Image.Annotations.Count(a => a.ClassId == classId) + 1}" : target.DisplayName;
            target.ClassAssignmentSource = ClassAssignmentSource.Manual;
            target.GeometrySource = GeometrySource.Segmentation;
            target.SegmentationAlgorithm = result.AlgorithmName;
            target.AlgorithmVersion = result.AlgorithmVersion;
            target.AlgorithmParameters = new Dictionary<string, string>(result.ParametersUsed)
            {
                ["efficiency.interactions"] = c.Interactions.ToString(CultureInfo.InvariantCulture),
                ["efficiency.refinementRuns"] = c.Runs.ToString(CultureInfo.InvariantCulture),
                ["efficiency.seconds"] = (Now - c.StartedAt).TotalSeconds.ToString("0.0", CultureInfo.InvariantCulture),
                ["efficiency.firstProposalRetained"] = c.FirstProposal is { IsEmpty: false } fp
                    ? ((double)fp.OverlapCount(mask) / fp.PixelCount).ToString("0.0000", CultureInfo.InvariantCulture) : "",
                ["roi"] = c.Roi.ToString(),
            };
            target.VerificationStatus = verification;
            if (!string.IsNullOrEmpty(notes)) target.Notes = notes;
            target.ModifiedAt = Now;
            if (editing is null) { Image.Annotations.Add(target); created = target; }
            else created = editing;

            if (overlap > 0 && pol == OverlapPolicy.ReplaceExisting)
                foreach (var o in others.Where(o => o.Mask.OverlapCount(mask) > 0))
                {
                    var rest = o.Mask.Subtract(mask);
                    if (rest.IsEmpty) { Image.Annotations.Remove(o); removed.Add(o.Id); }
                    else
                    {
                        PushRevision(o, $"Overlap replaced by {target.Id}");
                        AnnotationGeometryService.SetMask(o, rest);
                        o.Revision++; o.ModifiedAt = Now; o.GeometrySource = GeometrySource.BooleanOperation;
                        modified.Add(o.Id);
                    }
                }
            if (Image.WorkflowState == ImageWorkflowState.Unannotated) Image.WorkflowState = ImageWorkflowState.InProgress;
        });

        History.Execute(new CompositeCommand(cmd.Description, new IUndoableCommand[]
        {
            cmd,
            new DelegateUndoableCommand("clear candidate", () => { Candidate = null; _candidateVersion++; }, () => { Candidate = c; _candidateVersion++; }),
        }));
        CandidateChanged?.Invoke(this, EventArgs.Empty);
        AnnotationsChanged?.Invoke(this, EventArgs.Empty);

        Log(UserActor, "Segmentation Accepted", created!.Id,
            $"{created.PixelArea} px; overlap {overlap} px resolved with {pol}" +
            (modified.Count + removed.Count > 0 ? $"; modified {modified.Count}, removed {removed.Count}" : ""));
        Log(UserActor, "Class Assigned", created.Id, $"Class: {cls.Name} ({cls.Id}); Verification: {verification}");
        return new AcceptResult(created, overlap, modified, removed);
    }

    // ───────────────────────── Edits on accepted annotations ─────────────────────────

    public Annotation Find(Guid id) => Image.Annotations.FirstOrDefault(a => a.Id == id)
        ?? throw new KeyNotFoundException($"Annotation {id} not found on this image.");

    private void PushRevision(Annotation a, string reason)
    {
        a.RevisionHistory.Add(new AnnotationRevision
        {
            Revision = a.Revision, Timestamp = a.ModifiedAt == default ? Now : a.ModifiedAt, Author = User, Reason = reason,
            ClassId = a.ClassId, VerificationStatus = a.VerificationStatus, GeometrySource = a.GeometrySource, Mask = a.Mask, Notes = a.Notes,
        });
    }

    private void Edit(string description, Action mutation, Guid? annotationId, string details)
    {
        History.Execute(new ImageSnapshotCommand(Image, description, mutation));
        AnnotationsChanged?.Invoke(this, EventArgs.Empty);
        Log(UserActor, description, annotationId, details);
    }

    private void ModifyGeometry(Annotation a, RasterMask newMask, GeometrySource source, string reason)
    {
        if (newMask.IsEmpty) throw new InvalidOperationException("The edit would leave the annotation empty; delete it instead.");
        PushRevision(a, reason);
        AnnotationGeometryService.SetMask(a, newMask.ClipToImage(Width, Height));
        a.GeometrySource = source; a.Revision++; a.ModifiedAt = Now;
    }

    public void Relabel(Guid id, int classId)
    {
        var a = Find(id);
        var cls = Project.Taxonomy.Find(classId) ?? throw new ArgumentException($"Unknown class {classId}.");
        if (classId == Taxonomy.UnlabeledId) throw new ArgumentException("Class 0 cannot be assigned.");
        int old = a.ClassId;
        Edit("Class Assigned", () =>
        {
            PushRevision(a, "Relabel");
            a.ClassId = classId; a.ClassAssignmentSource = ClassAssignmentSource.Manual; a.Revision++; a.ModifiedAt = Now;
        }, id, $"Class: {old} → {cls.Name} ({classId}); verification unchanged ({a.VerificationStatus})");
    }

    /// <summary>Accepts the class suggestion; verification status is deliberately left unchanged.</summary>
    public void AcceptSuggestion(Guid id)
    {
        var a = Find(id);
        var s = a.Suggestion ?? throw new InvalidOperationException("No suggestion available.");
        Edit("Suggestion Accepted", () =>
        {
            PushRevision(a, "Accept suggestion");
            a.ClassId = s.ClassId; a.ClassAssignmentSource = ClassAssignmentSource.AcceptedSuggestion; a.Revision++; a.ModifiedAt = Now;
        }, id, $"Class {s.ClassId} (similarity {s.SimilarityScore:0.00}, uncalibrated); verification unchanged");
    }

    public void SetVerification(Guid id, VerificationStatus status, string? reviewer = null, EvidenceReference? evidence = null)
    {
        var a = Find(id);
        Edit("Verification Changed", () =>
        {
            a.VerificationStatus = status;
            if (status >= VerificationStatus.ExpertReviewed) { a.ReviewedBy = reviewer ?? User; a.ReviewedAt = Now; }
            if (evidence is not null) a.Evidence.Add(evidence);
            a.ModifiedAt = Now;
        }, id, $"Verification: {status}" + (evidence is null ? "" : $"; evidence {evidence.Type}: {evidence.Source}"));
    }

    /// <summary>Non-geometric metadata edit (notes, severity, display name, reviewer confidence …).</summary>
    public void UpdateMetadata(Guid id, string description, Action<Annotation> change)
    {
        var a = Find(id);
        Edit(description, () => { change(a); a.ModifiedAt = Now; }, id, description);
    }

    public void SetVisibility(Guid id, bool visible) => UpdateMetadata(id, visible ? "Show annotation" : "Hide annotation", a => a.IsVisible = visible);

    public void Delete(IReadOnlyCollection<Guid> ids)
    {
        var targets = ids.Select(Find).ToList();
        Edit("Annotation Deleted", () => { foreach (var t in targets) Image.Annotations.Remove(t); }, targets.FirstOrDefault()?.Id,
            $"{targets.Count} annotation(s): {string.Join(", ", targets.Select(t => t.Id))}");
    }

    /// <summary>Applies an edited polygon geometry (vertex drag/add/delete, hole edit, move).</summary>
    public void EditPolygons(Guid id, IReadOnlyList<PolygonComponent> polygons)
    {
        var a = Find(id);
        Edit("Polygon Edited", () =>
        {
            PushRevision(a, "Polygon edit");
            AnnotationGeometryService.SetPolygons(a, polygons, Width, Height);
            if (a.Mask.IsEmpty) throw new InvalidOperationException("The edited polygon has no area inside the image.");
            a.GeometrySource = GeometrySource.PolygonEdit; a.Revision++; a.ModifiedAt = Now;
        }, id, $"{polygons.Sum(p => p.VertexCount)} vertices; mask re-rasterised");
    }

    public void BrushEdit(Guid id, Stroke stroke, bool add)
    {
        var a = Find(id);
        Edit(add ? "Brush Added" : "Brush Erased", () =>
            ModifyGeometry(a, SegmentationEngine.Paint(a.Mask, stroke, add, Width, Height), GeometrySource.BrushEdit, add ? "Brush add" : "Erase"),
            id, $"{stroke.Points.Count} points, radius {stroke.Radius}");
    }

    public void Move(Guid id, int dx, int dy)
    {
        var a = Find(id);
        Edit("Annotation Moved", () => ModifyGeometry(a, a.Mask.Translate(dx, dy), GeometrySource.PolygonEdit, "Move"), id, $"Δ=({dx}, {dy})");
    }

    public void FillHoles(Guid id)
    {
        var a = Find(id);
        Edit("Holes Filled", () => ModifyGeometry(a, SegmentationEngine.FillHoles(a.Mask), GeometrySource.BooleanOperation, "Fill holes"), id, "");
    }

    public enum BooleanOp { Union, Intersection, Subtraction }

    /// <summary>target := target ∘ operand; the operand annotation is left unchanged.</summary>
    public void Boolean(Guid targetId, Guid operandId, BooleanOp op)
    {
        var t = Find(targetId); var o = Find(operandId);
        var mask = op switch
        {
            BooleanOp.Union => t.Mask.Union(o.Mask),
            BooleanOp.Intersection => t.Mask.Intersect(o.Mask),
            _ => t.Mask.Subtract(o.Mask),
        };
        Edit($"Region {op}", () => ModifyGeometry(t, mask, GeometrySource.BooleanOperation, $"{op} with {operandId}"), targetId, $"operand {operandId}");
    }

    /// <summary>Merges annotations into the first one (union); the others are removed. Classes must match unless forced.</summary>
    public Annotation Merge(IReadOnlyList<Guid> ids, bool allowDifferentClasses = false)
    {
        if (ids.Count < 2) throw new ArgumentException("Select at least two annotations to merge.");
        var items = ids.Select(Find).ToList();
        if (!allowDifferentClasses && items.Select(i => i.ClassId).Distinct().Count() > 1)
            throw new InvalidOperationException("Cannot merge annotations of different classes.");
        var target = items[0];
        Edit("Regions Merged", () =>
        {
            var mask = items.Skip(1).Aggregate(target.Mask, (m, a) => m.Union(a.Mask));
            ModifyGeometry(target, mask, GeometrySource.Merge, $"Merged {ids.Count - 1} region(s)");
            foreach (var other in items.Skip(1)) Image.Annotations.Remove(other);
        }, target.Id, $"Merged {string.Join(", ", ids)}");
        return target;
    }

    /// <summary>Splits an annotation along a cut line; each resulting part becomes its own annotation with copied metadata.</summary>
    public IReadOnlyList<Annotation> Split(Guid id, IReadOnlyList<PixelPoint> cut)
    {
        var a = Find(id);
        var parts = SegmentationEngine.Split(a.Mask, cut);
        if (parts.Count < 2) throw new InvalidOperationException("The cut line does not separate the region into several parts.");
        var created = new List<Annotation>();
        Edit("Region Split", () =>
        {
            created.Clear();
            ModifyGeometry(a, parts[0], GeometrySource.Split, "Split (part 1)");
            created.Add(a);
            foreach (var part in parts.Skip(1))
            {
                var copy = a.DeepClone();
                copy.Id = Guid.NewGuid(); copy.CreatedAt = Now; copy.CreatedBy = User; copy.Revision = 1;
                copy.RevisionHistory = new(); copy.ZOrder = Image.Annotations.Max(x => x.ZOrder) + 1;
                AnnotationGeometryService.SetMask(copy, part);
                copy.GeometrySource = GeometrySource.Split;
                Image.Annotations.Add(copy);
                created.Add(copy);
            }
        }, id, $"{parts.Count} parts");
        return created;
    }

    public Annotation Duplicate(Guid id, int dx = 10, int dy = 10)
    {
        var a = Find(id);
        Annotation? copy = null;
        Edit("Annotation Duplicated", () =>
        {
            copy = a.DeepClone();
            copy.Id = Guid.NewGuid(); copy.CreatedAt = Now; copy.ModifiedAt = Now; copy.CreatedBy = User;
            copy.Revision = 1; copy.RevisionHistory = new(); copy.ZOrder = Image.Annotations.Max(x => x.ZOrder) + 1;
            copy.VerificationStatus = VerificationStatus.Unverified; copy.Evidence = new(); copy.ReviewedBy = null; copy.ReviewedAt = null;
            AnnotationGeometryService.SetMask(copy, a.Mask.Translate(dx, dy).ClipToImage(Width, Height));
            Image.Annotations.Add(copy);
        }, id, $"offset ({dx}, {dy})");
        return copy!;
    }

    /// <summary>Restores an earlier revision's geometry, class and verification as a new revision.</summary>
    public void RestoreRevision(Guid id, int revision)
    {
        var a = Find(id);
        var r = a.RevisionHistory.FirstOrDefault(x => x.Revision == revision) ?? throw new KeyNotFoundException($"Revision {revision} not found.");
        Edit("Revision Restored", () =>
        {
            PushRevision(a, $"Restore revision {revision}");
            AnnotationGeometryService.SetMask(a, r.Mask);
            a.ClassId = r.ClassId; a.VerificationStatus = r.VerificationStatus; a.GeometrySource = r.GeometrySource;
            a.Revision++; a.ModifiedAt = Now;
        }, id, $"revision {revision}");
    }

    public void SetWorkflowState(ImageWorkflowState state)
    {
        var old = Image.WorkflowState;
        Edit("Workflow State Changed", () => Image.WorkflowState = state, null, $"{old} → {state}");
    }

    /// <summary>Directly accepts a mask produced outside the candidate workflow (e.g. a region suggestion or batch run).</summary>
    public AcceptResult AcceptMask(RasterMask mask, int classId, VerificationStatus verification, string algorithm,
        Dictionary<string, string>? parameters = null, OverlapPolicy? policy = null)
    {
        SetCandidate(new CandidateState
        {
            Roi = mask.Bounds.ClampTo(Width, Height), Parameters = new SegmentationParameters(), StartedAt = Now,
            Result = new SegmentationResult
            {
                AlgorithmName = algorithm, AlgorithmVersion = "1.0.0", Mask = mask, Components = Vision.Contours.ContourExtractor.Extract(mask),
                ProcessingTime = TimeSpan.Zero, ParametersUsed = parameters ?? new(),
            },
            FirstProposal = mask,
        }, "Load proposal", undoable: false);
        return AcceptCandidate(classId, verification, policy: policy);
    }

    public void Undo() { History.Undo(); AnnotationsChanged?.Invoke(this, EventArgs.Empty); CandidateChanged?.Invoke(this, EventArgs.Empty); }
    public void Redo() { History.Redo(); AnnotationsChanged?.Invoke(this, EventArgs.Empty); CandidateChanged?.Invoke(this, EventArgs.Empty); }
}
