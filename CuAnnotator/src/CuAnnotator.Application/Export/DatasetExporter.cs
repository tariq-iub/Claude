using System.Globalization;
using System.Text;
using System.Text.Json;
using CuAnnotator.Application.Services;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Vision.ImageIO;

namespace CuAnnotator.Application.Export;

public sealed class ExportReport
{
    public List<string> Warnings { get; } = new();
    public List<string> Messages { get; } = new();
    public List<string> FilesWritten { get; } = new();
    public void Warn(string s) { lock (Warnings) Warnings.Add(s); }
    public void Info(string s) { lock (Messages) Messages.Add(s); }
}

public sealed class DatasetExportOptions
{
    public required string OutputFolder { get; init; }
    public bool IncludeImages { get; init; } = true;
    /// <summary>Re-encode images losslessly as PNG (true) or copy the original files byte-for-byte (false).</summary>
    public bool ImagesAsPng { get; init; } = true;
    public bool IncludeSemanticMasks { get; init; } = true;
    public bool IncludeColorizedMasks { get; init; } = true;
    public bool IncludeNativeJson { get; init; } = true;
    public bool IncludeCoco { get; init; } = true;
    public CocoSegmentationMode CocoMode { get; init; } = CocoSegmentationMode.Auto;
    public bool IncludeYolo { get; init; }
    public YoloHolePolicy YoloHoles { get; init; } = YoloHolePolicy.Skip;
    /// <summary>Optional per-class binary masks (class IDs). Empty = none.</summary>
    public IReadOnlyCollection<int> BinaryMaskClasses { get; init; } = Array.Empty<int>();
    /// <summary>Also export one binary mask of all corrosion classes (every class except Healthy, Ignore).</summary>
    public bool IncludeAllCorrosionBinary { get; init; }
    public bool OnlyAnnotatedImages { get; init; } = true;
    public bool CreateSplits { get; init; } = true;
    public double TrainFraction { get; init; } = 0.7;
    public double ValidationFraction { get; init; } = 0.15;
    public int SplitSeed { get; init; } = 42;
    public bool GroupBySpecimen { get; init; } = true;
    public string AnnotationSet { get; init; } = "default";
}

/// <summary>Writes the ML dataset directory described in 09_Export_Formats.md and validates it.</summary>
public sealed class DatasetExporter(IProjectSerializer serializer)
{
    public const int HealthyCopperId = 1;

    public async Task<ExportReport> ExportAsync(AnnotationProject project, string projectFolder, DatasetExportOptions o,
        IProgress<(int Done, int Total, string Item)>? progress = null, CancellationToken ct = default)
    {
        var report = new ExportReport();
        var root = Path.GetFullPath(o.OutputFolder);
        Directory.CreateDirectory(root);
        var images = project.Images.Where(i => !o.OnlyAnnotatedImages || i.Annotations.Any(a => a.AnnotationSet == o.AnnotationSet)).ToList();
        if (images.Count == 0) report.Warn("No annotated images to export.");

        var split = o.CreateSplits
            ? DatasetSplitter.Split(images, o.TrainFraction, o.ValidationFraction, o.SplitSeed, o.GroupBySpecimen)
            : images.ToDictionary(i => i.Id, _ => DatasetSplit.Train);
        var leaks = DatasetSplitter.LeakingSpecimens(images, split);
        foreach (var l in leaks) report.Warn($"Specimen '{l}' appears in several splits (grouping disabled).");

        var names = UniqueStems(images);
        var palette = project.Taxonomy.Classes.ToDictionary(c => c.Id, c => ParseColor(c.ColorHex));
        var yoloMap = YoloExporter.ClassIndexMap(project.Taxonomy);
        var imageFileNames = new Dictionary<Guid, string>();
        int done = 0;

        await Task.Run(() =>
        {
            foreach (var img in images)
            {
                ct.ThrowIfCancellationRequested();
                var stem = names[img.Id];
                var src = ProjectPaths.Resolve(projectFolder, img.RelativePath);
                string imageFile = o.ImagesAsPng ? stem + ".png" : stem + Path.GetExtension(img.RelativePath).ToLowerInvariant();
                imageFileNames[img.Id] = imageFile;

                if (o.IncludeImages)
                {
                    var dst = Path.Combine(root, "images", imageFile);
                    if (!File.Exists(src)) report.Warn($"Source image missing: {src}");
                    else if (o.ImagesAsPng)
                    {
                        using var li = ImageLoader.Load(src);
                        if (!string.IsNullOrEmpty(img.Sha256) && li.Info.Sha256 != img.Sha256) report.Warn($"{img.FileName}: checksum differs from the one recorded at import.");
                        MaskIO.WriteImagePng(dst, li.Bgr);
                    }
                    else
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
                        File.Copy(src, dst, overwrite: true);
                    }
                    report.FilesWritten.Add(dst);
                }

                var map = LabelMapComposer.Compose(img, o.AnnotationSet);
                if (map.OverlapPixels > 0) report.Warn($"{img.FileName}: {map.OverlapPixels} overlapping pixels resolved by Z-order (topmost wins).");
                if (o.IncludeSemanticMasks)
                {
                    var p = Path.Combine(root, "masks", stem + ".png");
                    MaskIO.WriteLabelPng(p, map.Labels, map.Width, map.Height);
                    report.FilesWritten.Add(p);
                }
                if (o.IncludeColorizedMasks)
                {
                    var p = Path.Combine(root, "colorized_masks", stem + ".png");
                    MaskIO.WriteColorizedPng(p, map.Labels, map.Width, map.Height, palette);
                    report.FilesWritten.Add(p);
                }
                foreach (var cls in o.BinaryMaskClasses)
                {
                    var p = Path.Combine(root, "binary_masks", $"class_{cls}", stem + ".png");
                    MaskIO.WriteLabelPng(p, LabelMapComposer.Binary(img, a => a.ClassId == cls, o.AnnotationSet), img.Width, img.Height);
                    report.FilesWritten.Add(p);
                }
                if (o.IncludeAllCorrosionBinary)
                {
                    var p = Path.Combine(root, "binary_masks", "all_corrosion", stem + ".png");
                    MaskIO.WriteLabelPng(p, LabelMapComposer.Binary(img,
                        a => a.ClassId is not (HealthyCopperId or Taxonomy.IgnoreId or Taxonomy.UnlabeledId), o.AnnotationSet), img.Width, img.Height);
                    report.FilesWritten.Add(p);
                }
                if (o.IncludeNativeJson)
                {
                    var p = Path.Combine(root, "annotations", stem + ".json");
                    Directory.CreateDirectory(Path.GetDirectoryName(p)!);
                    File.WriteAllText(p, serializer.SerializeImage(project, img));
                    report.FilesWritten.Add(p);
                }
                if (o.IncludeYolo)
                {
                    var s = SplitFolder(split[img.Id]);
                    var p = Path.Combine(root, "yolo", "labels", s, stem + ".txt");
                    Directory.CreateDirectory(Path.GetDirectoryName(p)!);
                    File.WriteAllText(p, YoloExporter.LabelFile(img, yoloMap, o.YoloHoles, report, o.AnnotationSet));
                    if (o.IncludeImages)
                    {
                        var ip = Path.Combine(root, "yolo", "images", s, imageFile);
                        Directory.CreateDirectory(Path.GetDirectoryName(ip)!);
                        File.Copy(Path.Combine(root, "images", imageFile), ip, true);
                    }
                    report.FilesWritten.Add(p);
                }
                progress?.Report((++done, images.Count, img.FileName));
            }
        }, ct);

        var json = new JsonSerializerOptions { WriteIndented = true };
        if (o.IncludeCoco)
        {
            var coco = CocoExporter.Build(project, images.Select(i => (i, imageFileNames[i.Id])), o.CocoMode, report, o.AnnotationSet);
            Write(root, "coco/annotations.json", JsonSerializer.Serialize(coco, json), report);
            foreach (var s in new[] { DatasetSplit.Train, DatasetSplit.Validation, DatasetSplit.Test })
            {
                var subset = images.Where(i => split[i.Id] == s).ToList();
                if (subset.Count == 0 || !o.CreateSplits) continue;
                var cs = CocoExporter.Build(project, subset.Select(i => (i, imageFileNames[i.Id])), o.CocoMode, new ExportReport(), o.AnnotationSet);
                Write(root, $"coco/annotations_{SplitFolder(s)}.json", JsonSerializer.Serialize(cs, json), report);
            }
        }
        if (o.IncludeYolo) Write(root, "yolo/data.yaml", YoloExporter.DataYaml(project.Taxonomy, root), report);

        Write(root, "metadata/classes.json", JsonSerializer.Serialize(project.Taxonomy.Classes.Select(c => new
        {
            id = c.Id, name = c.Name, formula = c.ChemicalFormula, description = c.Description, color = c.ColorHex,
            parent = c.ParentCategory, enabled = c.IsEnabled, reserved = c.IsReserved, requiredVerification = c.RequiredVerification.ToString(),
            yoloIndex = yoloMap.TryGetValue(c.Id, out var yi) ? yi : (int?)null,
        }), json), report);
        Write(root, "metadata/dataset.json", JsonSerializer.Serialize(new
        {
            format = "CuAnnotator dataset", project = project.Name, projectId = project.Id, exportedAt = DateTimeOffset.UtcNow.ToString("O"),
            appVersion = AppInfo.Version, taxonomyVersion = project.Taxonomy.Version, coordinateConvention = Contracts.NativeFormat.CoordinateConvention,
            maskEncoding = "8-bit single-channel PNG; pixel value = class ID; 0 = unlabeled; 255 = ignore",
            split = new { seed = o.SplitSeed, train = o.TrainFraction, validation = o.ValidationFraction, groupedBySpecimen = o.GroupBySpecimen, enabled = o.CreateSplits },
            images = images.Select(i => new
            {
                file = imageFileNames[i.Id], mask = names[i.Id] + ".png", imageId = i.Id, original = i.FileName, sha256 = i.Sha256,
                width = i.Width, height = i.Height, specimen = i.SpecimenId, split = SplitFolder(split[i.Id]), workflow = i.WorkflowState.ToString(),
                micrometersPerPixel = i.MicrometersPerPixel, magnification = i.Magnification,
            }),
        }, json), report);
        Write(root, "metadata/verification.json", JsonSerializer.Serialize(images.SelectMany(i => i.Annotations.Where(a => a.AnnotationSet == o.AnnotationSet).Select(a => new
        {
            image = imageFileNames[i.Id], annotationId = a.Id, classId = a.ClassId, verification = a.VerificationStatus.ToString(),
            classAssignment = a.ClassAssignmentSource.ToString(), reviewedBy = a.ReviewedBy,
            evidence = a.Evidence.Select(e => new { type = e.Type.ToString(), e.Source, e.Notes }),
        })), json), report);
        Write(root, "README.md", Readme(project, images.Count, report), report);

        foreach (var w in Validate(root, images, names, o)) report.Warn(w);
        return report;
    }

    /// <summary>Re-reads every exported mask and checks dimensions and class values against the project.</summary>
    public static IEnumerable<string> Validate(string root, IReadOnlyList<ImageItem> images, IReadOnlyDictionary<Guid, string> names, DatasetExportOptions o)
    {
        if (!o.IncludeSemanticMasks) yield break;
        foreach (var img in images)
        {
            var path = Path.Combine(root, "masks", names[img.Id] + ".png");
            var (labels, w, h) = MaskIO.ReadLabelPng(path);
            if (w != img.Width || h != img.Height) { yield return $"{path}: {w}×{h} does not match source {img.Width}×{img.Height}."; continue; }
            var expected = LabelMapComposer.Compose(img, o.AnnotationSet).Labels;
            if (!labels.AsSpan().SequenceEqual(expected)) yield return $"{path}: pixel values differ from the project annotations.";
        }
    }

    private static Dictionary<Guid, string> UniqueStems(IEnumerable<ImageItem> images)
    {
        var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new Dictionary<Guid, string>();
        foreach (var i in images)
        {
            var baseName = string.Concat(Path.GetFileNameWithoutExtension(i.FileName).Select(ch => char.IsLetterOrDigit(ch) || ch is '_' or '-' ? ch : '_'));
            var name = baseName; int k = 2;
            while (!used.Add(name)) name = $"{baseName}_{k++}";
            result[i.Id] = name;
        }
        return result;
    }

    private static string SplitFolder(DatasetSplit s) => s switch { DatasetSplit.Train => "train", DatasetSplit.Validation => "val", _ => "test" };

    private static void Write(string root, string rel, string content, ExportReport report)
    {
        var p = Path.Combine(root, rel);
        Directory.CreateDirectory(Path.GetDirectoryName(p)!);
        File.WriteAllText(p, content, new UTF8Encoding(false));
        report.FilesWritten.Add(p);
    }

    public static (byte R, byte G, byte B) ParseColor(string hex)
    {
        hex = hex.TrimStart('#');
        return (byte.Parse(hex[..2], NumberStyles.HexNumber), byte.Parse(hex[2..4], NumberStyles.HexNumber), byte.Parse(hex[4..6], NumberStyles.HexNumber));
    }

    private static string Readme(AnnotationProject p, int count, ExportReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# {p.Name} — CuAnnotator dataset export").AppendLine();
        sb.AppendLine($"Exported {DateTimeOffset.UtcNow:yyyy-MM-dd HH:mm} UTC with CuAnnotator {AppInfo.Version}; {count} image(s).").AppendLine();
        sb.AppendLine("## Layout").AppendLine();
        sb.AppendLine("- `images/` source images (lossless PNG or byte-identical copies)");
        sb.AppendLine("- `masks/` semantic masks: 8-bit single-channel PNG, same size as the image, pixel value = class ID (0 unlabeled, 255 ignore)");
        sb.AppendLine("- `colorized_masks/` RGB visualisation only — do not train on these");
        sb.AppendLine("- `annotations/` native per-image JSON (lossless, re-editable)");
        sb.AppendLine("- `coco/` COCO instances (polygons, or compressed RLE when holes are present)");
        sb.AppendLine("- `yolo/` YOLO segmentation labels (see warnings for holes / multipart regions)");
        sb.AppendLine("- `metadata/` classes, dataset split and checksums, verification status").AppendLine();
        sb.AppendLine("## Classes").AppendLine().AppendLine("| ID | Name | Formula |").AppendLine("|---|---|---|");
        foreach (var c in p.Taxonomy.Classes.OrderBy(c => c.Id)) sb.AppendLine($"| {c.Id} | {c.Name} | {c.ChemicalFormula} |");
        sb.AppendLine().AppendLine("## Scientific note").AppendLine();
        sb.AppendLine("Class labels are annotations, not chemical measurements. Check `metadata/verification.json`: only annotations marked " +
                      "`InstrumentallyVerified` are backed by recorded analytical evidence.");
        if (report.Warnings.Count > 0)
        {
            sb.AppendLine().AppendLine("## Export warnings").AppendLine();
            foreach (var w in report.Warnings) sb.AppendLine($"- {w}");
        }
        return sb.ToString();
    }
}
