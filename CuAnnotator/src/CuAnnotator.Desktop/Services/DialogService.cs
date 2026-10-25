using System.IO;
using System.Windows;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Desktop.Views;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Desktop.Services;

public interface IDialogService
{
    string? PickFolder(string title);
    string[] OpenFiles(string filter);
    string? SaveFile(string filter, string defaultName);
    bool Confirm(string message);
    bool? AskSaveChanges();
    void ShowError(string message);
    void ShowWarning(string message);
    void ShowInfo(string message);
    void ShowText(string title, string text, IReadOnlyDictionary<string, Func<string>>? exports);
    NewProjectRequest? ShowNewProject(string userName);
    (string? Specimen, string? Source, double? Magnification, double? MicrometersPerPixel)? AskImportMetadata();
    DatasetExportOptions? ShowExport(AnnotationProject project, string defaultFolder);
    Taxonomy? ShowTaxonomyEditor(Taxonomy taxonomy, IReadOnlyCollection<int> usedIds);
    (Guid ImageId, Guid AnnotationId)? ShowReviewQueue(AnnotationProject project);
    OverlapPolicy? AskOverlapPolicy(long overlapPixels);
    (VerificationStatus Status, EvidenceReference Evidence)? AskEvidence();
    bool EditImageMetadata(ImageItem image);
    bool ShowProjectSettings(AnnotationProject project, UserSettings settings);
    void ShowHelp();
    void ShowOnboarding();
}

public sealed class DialogService(UserSettings settings) : IDialogService
{
    private static Window? Owner => System.Windows.Application.Current.MainWindow is { IsVisible: true } w ? w : null;
    private const string Caption = "CuAnnotator";

    public string? PickFolder(string title)
    {
        var d = new Microsoft.Win32.OpenFolderDialog { Title = title };
        return d.ShowDialog(Owner) == true ? d.FolderName : null;
    }

    public string[] OpenFiles(string filter)
    {
        var d = new Microsoft.Win32.OpenFileDialog { Filter = filter, Multiselect = true };
        return d.ShowDialog(Owner) == true ? d.FileNames : Array.Empty<string>();
    }

    public string? SaveFile(string filter, string defaultName)
    {
        var d = new Microsoft.Win32.SaveFileDialog { Filter = filter, FileName = defaultName };
        return d.ShowDialog(Owner) == true ? d.FileName : null;
    }

    public bool Confirm(string message) =>
        MessageBox.Show(Owner!, message, Caption, MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes;

    public bool? AskSaveChanges() =>
        MessageBox.Show(Owner!, "The project has unsaved changes. Save them now?", Caption, MessageBoxButton.YesNoCancel, MessageBoxImage.Warning) switch
        {
            MessageBoxResult.Yes => true,
            MessageBoxResult.No => false,
            _ => null,
        };

    public void ShowError(string message) => MessageBox.Show(Owner!, message, Caption, MessageBoxButton.OK, MessageBoxImage.Error);
    public void ShowWarning(string message) => MessageBox.Show(Owner!, message, Caption, MessageBoxButton.OK, MessageBoxImage.Warning);
    public void ShowInfo(string message) => MessageBox.Show(Owner!, message, Caption, MessageBoxButton.OK, MessageBoxImage.Information);

    public void ShowText(string title, string text, IReadOnlyDictionary<string, Func<string>>? exports) =>
        new TextReportWindow(title, text, exports).ShowDialog();

    public NewProjectRequest? ShowNewProject(string userName)
    {
        var f = new FormWindow("New annotation project", "A project folder holds the project file, a SQLite index, the audit trail and (by default) copies of imported images.")
            .Text("name", "Project name", required: true)
            .Folder("folder", "Project folder (an empty or new folder)")
            .Text("researcher", "Researcher", userName)
            .Text("institution", "Institution / laboratory")
            .Text("description", "Description", multiline: true)
            .Choice("algorithm", "Default segmentation algorithm", Enum.GetValues<SegmentationAlgorithmKind>().Take(5), SegmentationAlgorithmKind.GrabCut)
            .Choice("overlap", "Overlap policy for new regions", Enum.GetValues<OverlapPolicy>(), OverlapPolicy.PreserveExisting)
            .Check("copy", "Copy imported images into the project folder (originals are never modified)", true)
            .Note("The default taxonomy (IDs 0–8, 255) is compatible with the original five-class copper-corrosion dataset and can be edited later.");
        if (!f.Run()) return null;
        var name = f.GetText("name");
        var folder = f.GetText("folder");
        if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any())
            folder = Path.Combine(folder, string.Concat(name.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c)));
        return new NewProjectRequest
        {
            Name = name, Folder = folder, Researcher = f.GetText("researcher"), Institution = f.GetText("institution"), Description = f.GetText("description"),
            Settings = new ProjectSettings
            {
                DefaultAlgorithm = f.Get<SegmentationAlgorithmKind>("algorithm"), OverlapPolicy = f.Get<OverlapPolicy>("overlap"),
                CopyImagesIntoProject = f.GetBool("copy"),
            },
        };
    }

    public (string? Specimen, string? Source, double? Magnification, double? MicrometersPerPixel)? AskImportMetadata()
    {
        var f = new FormWindow("Image metadata", "Optional metadata applied to all selected images. Images sharing a specimen ID are kept in the same dataset split.")
            .Text("specimen", "Specimen identifier")
            .Choice("source", "Acquisition source", new[] { "", "Digital camera", "Laboratory microscope", "Optical inspection system", "Industrial camera", "Flatbed scanner", "Scientific dataset" }, "")
            .Number("mag", "Magnification (×)", null, 0, 1e6, optional: true)
            .Number("mpp", "Pixel size (µm per pixel)", null, 0, 1e9, optional: true);
        if (!f.Run()) return null;
        string? N(string s) => string.IsNullOrWhiteSpace(s) ? null : s;
        return (N(f.GetText("specimen")), N(f.Get<string>("source")), f.GetNumber("mag"), f.GetNumber("mpp"));
    }

    public bool EditImageMetadata(ImageItem image)
    {
        var f = new FormWindow($"Metadata — {image.FileName}",
                $"{image.Width}×{image.Height} {image.PixelFormat}, {image.FileSizeBytes:N0} bytes, SHA-256 {image.Sha256[..Math.Min(16, image.Sha256.Length)]}…\n" +
                $"Colour profile: {image.ColorProfile}. EXIF orientation: {image.ExifOrientation?.ToString() ?? "none"}{(image.ExifOrientationApplied ? " (applied on import)" : "")}.")
            .Text("specimen", "Specimen identifier", image.SpecimenId ?? "")
            .Text("source", "Acquisition source", image.AcquisitionSource ?? "")
            .Number("mag", "Magnification (×)", image.Magnification, 0, 1e6, optional: true)
            .Number("mpp", "Pixel size (µm per pixel)", image.MicrometersPerPixel, 0, 1e9, optional: true)
            .Text("notes", "Notes", image.Notes, multiline: true);
        if (!f.Run()) return false;
        image.SpecimenId = string.IsNullOrWhiteSpace(f.GetText("specimen")) ? null : f.GetText("specimen");
        image.AcquisitionSource = string.IsNullOrWhiteSpace(f.GetText("source")) ? null : f.GetText("source");
        image.Magnification = f.GetNumber("mag");
        image.MicrometersPerPixel = f.GetNumber("mpp");
        image.Notes = f.GetText("notes");
        return true;
    }

    public DatasetExportOptions? ShowExport(AnnotationProject project, string defaultFolder)
    {
        var e = project.Settings.Export;
        var f = new FormWindow("Export dataset", "Semantic masks are 8-bit single-channel PNGs with class IDs as pixel values, at the exact size of each source image.", 520)
            .Folder("folder", "Output folder", defaultFolder)
            .Choice("type", "Export type", new[] { "Semantic segmentation", "Instance segmentation", "Class-specific binary", "Image summaries only" }, "Semantic segmentation")
            .Check("images", "Include images", true)
            .Check("png", "Re-encode images as lossless PNG (otherwise byte-identical copies)", e.CopyImagesAsPng)
            .Check("color", "Include colourised masks (visual inspection only)", e.IncludeColorizedMasks)
            .Check("native", "Include native per-image JSON (lossless, re-editable)", e.IncludeNativeJson)
            .Check("coco", "Include COCO instances", e.IncludeCoco)
            .Check("rle", "COCO: always use RLE (pixel-exact) instead of polygons where possible", false)
            .Check("yolo", "Include YOLO segmentation labels", e.IncludeYolo)
            .Check("yoloOuter", "YOLO: export outer boundary of regions with holes (otherwise skip them)", false)
            .Check("allCorrosion", "Binary mask of all corrosion classes", false)
            .Text("binaryClasses", "Binary masks for class IDs (comma-separated, optional)", "")
            .Check("annotatedOnly", "Only images with annotations", true)
            .Check("split", "Create train / validation / test split", true)
            .Number("train", "Train fraction", e.TrainFraction, 0, 1)
            .Number("val", "Validation fraction", e.ValidationFraction, 0, 1)
            .Number("seed", "Split random seed", e.SplitSeed, int.MinValue, int.MaxValue)
            .Check("group", "Keep images of the same specimen in the same split (prevents leakage)", e.GroupBySpecimen);
        if (!f.Run()) return null;
        var type = f.Get<string>("type");
        var binary = f.GetText("binaryClasses").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => int.TryParse(s, out var v) ? v : -1).Where(v => v is >= 0 and <= 255).ToList();
        bool summaries = type == "Image summaries only";
        e.TrainFraction = f.GetNumber("train") ?? 0.7; e.ValidationFraction = f.GetNumber("val") ?? 0.15;
        e.SplitSeed = (int)(f.GetNumber("seed") ?? 42); e.GroupBySpecimen = f.GetBool("group");
        return new DatasetExportOptions
        {
            OutputFolder = f.GetText("folder"),
            IncludeImages = f.GetBool("images") && !summaries,
            ImagesAsPng = f.GetBool("png"),
            IncludeSemanticMasks = type is "Semantic segmentation" or "Instance segmentation" or "Class-specific binary",
            IncludeColorizedMasks = f.GetBool("color") && !summaries,
            IncludeNativeJson = f.GetBool("native") || summaries,
            IncludeCoco = f.GetBool("coco") || type == "Instance segmentation",
            CocoMode = f.GetBool("rle") ? CocoSegmentationMode.AlwaysRle : CocoSegmentationMode.Auto,
            IncludeYolo = f.GetBool("yolo"),
            YoloHoles = f.GetBool("yoloOuter") ? YoloHolePolicy.OuterBoundaryOnly : YoloHolePolicy.Skip,
            BinaryMaskClasses = type == "Class-specific binary" && binary.Count == 0 ? project.Taxonomy.AssignableClasses.Select(c => c.Id).ToList() : binary,
            IncludeAllCorrosionBinary = f.GetBool("allCorrosion"),
            OnlyAnnotatedImages = f.GetBool("annotatedOnly"),
            CreateSplits = f.GetBool("split"),
            TrainFraction = e.TrainFraction, ValidationFraction = e.ValidationFraction, SplitSeed = e.SplitSeed, GroupBySpecimen = e.GroupBySpecimen,
        };
    }

    public Taxonomy? ShowTaxonomyEditor(Taxonomy taxonomy, IReadOnlyCollection<int> usedIds)
    {
        var w = new TaxonomyEditorWindow(taxonomy, usedIds);
        return w.ShowDialog() == true ? w.Result : null;
    }

    public (Guid ImageId, Guid AnnotationId)? ShowReviewQueue(AnnotationProject project)
    {
        var w = new ReviewQueueWindow(project);
        return w.ShowDialog() == true ? w.Result : null;
    }

    public OverlapPolicy? AskOverlapPolicy(long overlapPixels)
    {
        var f = new FormWindow("Overlapping annotation", $"The new region overlaps existing annotations by {overlapPixels:N0} pixels. How should the overlap be resolved?")
            .Choice("p", "Resolution", new[] { OverlapPolicy.PreserveExisting, OverlapPolicy.ReplaceExisting, OverlapPolicy.AllowOverlapTopmostWins }, OverlapPolicy.PreserveExisting)
            .Note("PreserveExisting: the new region loses the overlapping pixels. ReplaceExisting: existing regions lose them (a revision is recorded). " +
                  "AllowOverlap: both keep them; the topmost wins in semantic masks and QA reports the overlap.");
        return f.Run() ? f.Get<OverlapPolicy>("p") : null;
    }

    public (VerificationStatus Status, EvidenceReference Evidence)? AskEvidence()
    {
        var f = new FormWindow("Add verification evidence", "Record the evidence supporting the chemical-phase identity. Instrumental verification requires an analytical reference.")
            .Choice("status", "New verification status", Enum.GetValues<VerificationStatus>(), VerificationStatus.ExpertReviewed)
            .Choice("type", "Evidence type", Enum.GetValues<EvidenceType>(), EvidenceType.Raman)
            .Text("source", "Source (run ID, file, DOI, notebook page …)", required: true)
            .Text("notes", "Notes", multiline: true);
        if (!f.Run()) return null;
        var status = f.Get<VerificationStatus>("status");
        var type = f.Get<EvidenceType>("type");
        if (status == VerificationStatus.InstrumentallyVerified && type is EvidenceType.ExpertObservation or EvidenceType.LaboratoryNotes or EvidenceType.Other)
        {
            ShowWarning("'Instrumentally verified' needs instrumental evidence (XRD, Raman, FTIR, SEM-EDS, XPS) or a published reference. The status was set to 'Expert reviewed'.");
            status = VerificationStatus.ExpertReviewed;
        }
        return (status, new EvidenceReference { Type = type, Source = f.GetText("source"), Notes = f.GetText("notes"), Date = DateTimeOffset.UtcNow });
    }

    public bool ShowProjectSettings(AnnotationProject project, UserSettings user)
    {
        var s = project.Settings;
        var f = new FormWindow("Project & user settings")
            .Text("name", "Project name", project.Name, required: true)
            .Text("researcher", "Researcher", project.Researcher)
            .Text("desc", "Description", project.Description, multiline: true)
            .Choice("algorithm", "Default segmentation algorithm", Enum.GetValues<SegmentationAlgorithmKind>().Take(5), s.DefaultAlgorithm)
            .Choice("overlap", "Overlap policy", Enum.GetValues<OverlapPolicy>(), s.OverlapPolicy)
            .Number("autosave", "Autosave interval (seconds, 0 = off)", s.AutosaveIntervalSeconds, 0, 86400)
            .Text("severity", "Severity levels (comma-separated)", string.Join(", ", s.SeverityLevels))
            .Text("criteria", "Severity criteria (documentation)", s.SeverityCriteria, multiline: true)
            .Text("user", "Your name (recorded as annotator / reviewer)", user.UserName, required: true)
            .Number("snap", "Boundary brush snap radius (px)", user.SnapRadius, 1, 100)
            .Note($"Keyboard shortcuts are configurable in {UserSettings.FilePath} (\"KeyMap\").");
        if (!f.Run()) return false;
        project.Name = f.GetText("name"); project.Researcher = f.GetText("researcher"); project.Description = f.GetText("desc");
        s.DefaultAlgorithm = f.Get<SegmentationAlgorithmKind>("algorithm");
        s.OverlapPolicy = f.Get<OverlapPolicy>("overlap");
        s.AutosaveIntervalSeconds = (int)(f.GetNumber("autosave") ?? 120);
        s.SeverityLevels = f.GetText("severity").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        s.SeverityCriteria = f.GetText("criteria");
        user.UserName = f.GetText("user");
        user.SnapRadius = (int)(f.GetNumber("snap") ?? 8);
        user.Save();
        return true;
    }

    public void ShowHelp() => ShowText("CuAnnotator help", HelpText.Full(settings.KeyMap), null);

    public void ShowOnboarding()
    {
        ShowText("Welcome to CuAnnotator", HelpText.Onboarding, null);
        settings.ShowOnboarding = false;
        settings.Save();
    }
}

public static class HelpText
{
    public const string Onboarding = """
        WELCOME TO CUANNOTATOR — a semi-automated annotation studio for copper corrosion images.

        The core workflow takes about a minute per region:

          1. New Project (toolbar) → choose a folder.              Import Images → pick your micrographs/photos.
          2. Select an image in the bottom browser. Zoom with the mouse wheel, pan with Space+drag or the middle button.
          3. Press R and drag a LOOSE rectangle around a corrosion region. GrabCut proposes an irregular boundary (cyan).
          4. Refine: B paints pixels that must be included (green), N paints pixels that must be excluded (red).
             The segmentation re-runs automatically ("Live preview").
          5. Press Enter to accept with the active class, or press 1–8 / I to accept with that class directly.
          6. In the right panel set the verification status. Visual labels are NOT chemical verification — use
             "Add evidence" to record XRD/Raman/FTIR/SEM-EDS/XPS results.
          7. Ctrl+S saves. Export Dataset writes pixel-exact class-ID PNG masks, COCO, YOLO and native JSON.

        Everything is undoable (Ctrl+Z / Ctrl+Y) and every action is recorded in the audit trail.
        Open Help (F1 or the ? button) for all tools and shortcuts.
        """;

    public static string Full(IReadOnlyDictionary<string, string> keyMap) => Onboarding + """


        TOOLS
          Select / Pan        click selects (Ctrl+click multi-select), drag pans
          Rectangle ROI       loose box → automatic segmentation with the selected algorithm
          Polygon ROI         click vertices, double-click/Enter closes; polygon becomes the initial mask
          Freehand lasso      drag around the region; becomes the initial mask
          Smart contour       rectangle → hybrid GrabCut + CIELAB + edge refinement
          Foreground brush    must-include guidance (or add pixels to the selected annotation)
          Background brush    must-exclude guidance
          Boundary brush      trace near an edge; the path snaps to gradients and becomes a barrier
          Eraser              remove pixels from the selected annotation
          Vertex editor       drag / add (double-click edge) / delete (right-click) vertices; Ctrl+drag moves
          Split               draw a cut line through the selected annotation
          Merge               Ctrl+click several annotations, then Enter
          Fill region         fill holes of an annotation
          Colour selection    select all similar pixels (CIEDE2000 ≤ threshold)
          Magic wand          connected region of similar colour
          Measure / inspect   pixel values (RGB, HSV, CIELAB) and distances

        ALGORITHMS (Segmentation panel)
          GrabCut, CIELAB colour-guided (CIEDE2000), marker-controlled watershed, morphological active contours
          (Chan–Vese / geodesic) and the hybrid pipeline. Parameters are stored with every accepted annotation.

        SCIENTIFIC NOTES
          • Colour values assume sRGB; embedded ICC profiles are recorded but not applied.
          • Class suggestions are uncalibrated similarity scores, never probabilities, and never verification.
          • Minimum component area can be set to 0 to preserve microscopic structures.

        KEYBOARD SHORTCUTS (configurable in settings.json)

        """ + string.Join("\n", keyMap.OrderBy(k => k.Key).Select(k => $"  {k.Value,-16} {k.Key}")) + "\n  1–8             assign class (by class shortcut)\n  Space+drag      pan\n  Mouse wheel     zoom at cursor\n";
}
