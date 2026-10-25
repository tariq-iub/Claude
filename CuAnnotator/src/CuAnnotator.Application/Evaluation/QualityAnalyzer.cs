using System.Globalization;
using System.Text;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Services;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Application.Evaluation;

public enum IssueSeverity { Info, Warning, Error }

public sealed record QualityIssue(IssueSeverity Severity, Guid? ImageId, Guid? AnnotationId, string Message, string Suggestion);

public sealed record ClassDistribution(int ClassId, string Name, long Pixels, double PercentOfImagePixels, int Instances, double MeanArea, double MedianArea, long MinArea, long MaxArea);

public sealed class QualityReport
{
    public long TotalPixels { get; init; }
    public long LabeledPixels { get; init; }
    public long IgnoredPixels { get; init; }
    public long UnlabeledPixels { get; init; }
    /// <summary>Labeled (excluding Ignore) / total × 100.</summary>
    public double CoveragePercent => TotalPixels == 0 ? 0 : 100.0 * LabeledPixels / TotalPixels;
    public double IgnoredPercent => TotalPixels == 0 ? 0 : 100.0 * IgnoredPixels / TotalPixels;
    public List<ClassDistribution> Classes { get; init; } = new();
    public List<QualityIssue> Issues { get; init; } = new();
    public Dictionary<VerificationStatus, int> VerificationCounts { get; init; } = new();
    /// <summary>Region-size histogram: bucket upper bound (px, powers of 10) → instance count.</summary>
    public SortedDictionary<long, int> SizeHistogram { get; init; } = new();

    public string ToText()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Coverage: {CoveragePercent:0.00} % labeled, {IgnoredPercent:0.00} % ignored, {100 - CoveragePercent - IgnoredPercent:0.00} % unlabeled ({TotalPixels:N0} px)");
        sb.AppendLine();
        sb.AppendLine("Class distribution:");
        foreach (var c in Classes)
            sb.AppendLine($"  {c.ClassId,3} {c.Name,-24} {c.Pixels,12:N0} px  {c.PercentOfImagePixels,6:0.00} %  {c.Instances,4} inst.  mean {c.MeanArea:0} px  median {c.MedianArea:0} px");
        sb.AppendLine();
        sb.AppendLine("Verification: " + string.Join(", ", VerificationCounts.Select(kv => $"{kv.Key} {kv.Value}")));
        sb.AppendLine("Region sizes: " + string.Join(", ", SizeHistogram.Select(kv => $"≤{kv.Key:N0} px: {kv.Value}")));
        sb.AppendLine();
        sb.AppendLine($"Issues ({Issues.Count}):");
        foreach (var i in Issues) sb.AppendLine($"  [{i.Severity}] {i.Message} → {i.Suggestion}");
        return sb.ToString();
    }

    public string ClassesCsv()
    {
        var sb = new StringBuilder("class_id,name,pixels,percent,instances,mean_area,median_area,min_area,max_area\n");
        foreach (var c in Classes)
            sb.AppendLine(string.Join(',', c.ClassId, Csv(c.Name), c.Pixels, F(c.PercentOfImagePixels), c.Instances, F(c.MeanArea), F(c.MedianArea), c.MinArea, c.MaxArea));
        return sb.ToString();
    }

    internal static string F(double v) => v.ToString("0.####", CultureInfo.InvariantCulture);
    internal static string Csv(string s) => s.Contains(',') || s.Contains('"') ? $"\"{s.Replace("\"", "\"\"")}\"" : s;
}

/// <summary>Coverage, class distribution and validation checks for one image or a whole project.</summary>
public static class QualityAnalyzer
{
    public static QualityReport Analyze(AnnotationProject project, IEnumerable<ImageItem> images, string? projectFolder = null, string set = "default")
    {
        var issues = new List<QualityIssue>();
        long total = 0, labeled = 0, ignored = 0;
        var pixelsPerClass = new Dictionary<int, long>();
        var areas = new Dictionary<int, List<long>>();
        var verification = Enum.GetValues<VerificationStatus>().ToDictionary(v => v, _ => 0);
        var sizeHist = new SortedDictionary<long, int>();

        foreach (var img in images)
        {
            total += (long)img.Width * img.Height;
            if (projectFolder is not null && !File.Exists(ProjectPaths.Resolve(projectFolder, img.RelativePath)))
                issues.Add(new(IssueSeverity.Error, img.Id, null, $"{img.FileName}: image file not found at '{img.RelativePath}'.", "Relink or restore the image file."));
            var map = LabelMapComposer.Compose(img, set);
            foreach (var v in map.Labels)
            {
                if (v == Taxonomy.IgnoreId) ignored++;
                else if (v != Taxonomy.UnlabeledId) { labeled++; pixelsPerClass[v] = pixelsPerClass.GetValueOrDefault(v) + 1; }
            }
            if (map.OverlapPixels > 0 && project.Settings.OverlapPolicy != OverlapPolicy.AllowOverlapTopmostWins)
                issues.Add(new(IssueSeverity.Warning, img.Id, null, $"{img.FileName}: {map.OverlapPixels} px are claimed by several annotations.", "Resolve overlaps (subtract or relabel)."));

            foreach (var a in img.Annotations.Where(a => a.AnnotationSet == set))
            {
                verification[a.VerificationStatus]++;
                var cls = project.Taxonomy.Find(a.ClassId);
                if (cls is null) issues.Add(new(IssueSeverity.Error, img.Id, a.Id, $"{img.FileName}: annotation uses undefined class {a.ClassId}.", "Add the class to the taxonomy or relabel."));
                else if (!cls.IsEnabled) issues.Add(new(IssueSeverity.Info, img.Id, a.Id, $"{img.FileName}: annotation uses disabled class '{cls.Name}'.", "Re-enable or relabel."));
                if (a.Mask.IsEmpty) { issues.Add(new(IssueSeverity.Error, img.Id, a.Id, $"{img.FileName}: annotation {a.DisplayName} has an empty mask.", "Delete it.")); continue; }
                var b = a.Mask.ContentBounds();
                if (b.X < 0 || b.Y < 0 || b.Right > img.Width || b.Bottom > img.Height)
                    issues.Add(new(IssueSeverity.Error, img.Id, a.Id, $"{img.FileName}: annotation {a.DisplayName} extends outside the image {b}.", "Clip or redraw."));
                foreach (var comp in a.Polygons)
                {
                    if (comp.Outer.Concat(comp.Holes.SelectMany(h => h)).Any(p => p.X < 0 || p.Y < 0 || p.X >= img.Width || p.Y >= img.Height))
                        issues.Add(new(IssueSeverity.Error, img.Id, a.Id, $"{img.FileName}: {a.DisplayName} has polygon vertices out of bounds.", "Edit the polygon."));
                    if (a.GeometrySource == GeometrySource.PolygonEdit && comp.Outer.Count >= 4 && comp.Outer.Count <= 5000 && PolygonMath.IsSelfIntersecting(comp.Outer))
                        issues.Add(new(IssueSeverity.Warning, img.Id, a.Id, $"{img.FileName}: {a.DisplayName} has a self-intersecting edited polygon.", "Move the crossing vertices."));
                }
                if (cls is not null && a.VerificationStatus < cls.RequiredVerification && a.ClassId != Taxonomy.IgnoreId)
                    issues.Add(new(IssueSeverity.Info, img.Id, a.Id, $"{img.FileName}: {a.DisplayName} ({cls.Name}) is {a.VerificationStatus}; class expects {cls.RequiredVerification}.", "Review or add evidence."));
                (areas.TryGetValue(a.ClassId, out var l) ? l : areas[a.ClassId] = new()).Add(a.PixelArea);
                long bucket = 10;
                while (a.PixelArea > bucket) bucket *= 10;
                sizeHist[bucket] = sizeHist.GetValueOrDefault(bucket) + 1;
            }
        }

        var classes = project.Taxonomy.Classes.Where(c => c.Id != Taxonomy.UnlabeledId)
            .Where(c => pixelsPerClass.ContainsKey(c.Id) || areas.ContainsKey(c.Id)).OrderBy(c => c.Id).Select(c =>
            {
                var ar = areas.GetValueOrDefault(c.Id) ?? new List<long>();
                var sorted = ar.OrderBy(x => x).ToList();
                double median = sorted.Count == 0 ? 0 : sorted.Count % 2 == 1 ? sorted[sorted.Count / 2] : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2.0;
                long px = c.Id == Taxonomy.IgnoreId ? ignored : pixelsPerClass.GetValueOrDefault(c.Id);
                return new ClassDistribution(c.Id, c.Name, px, total == 0 ? 0 : 100.0 * px / total, ar.Count,
                    ar.Count == 0 ? 0 : ar.Average(), median, ar.Count == 0 ? 0 : ar.Min(), ar.Count == 0 ? 0 : ar.Max());
            }).ToList();

        return new QualityReport
        {
            TotalPixels = total, LabeledPixels = labeled, IgnoredPixels = ignored, UnlabeledPixels = total - labeled - ignored,
            Classes = classes, Issues = issues, VerificationCounts = verification, SizeHistogram = sizeHist,
        };
    }
}
