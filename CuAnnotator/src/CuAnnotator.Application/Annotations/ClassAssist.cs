using System.Globalization;
using System.Text;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.Contours;
using CuAnnotator.Vision.Interop;
using OpenCvSharp;

namespace CuAnnotator.Application.Annotations;

using Annotation = Domain.Entities.Annotation;

public sealed record ClassSuggestionResult(int ClassId, string ClassName, double SimilarityScore, double VoteShare, int NeighboursUsed, string Basis);

/// <summary>
/// Optional class-assisted annotation: distance-weighted k-nearest-neighbour vote over standardised
/// region descriptors (<see cref="RegionStatistics.ToDescriptor"/>) of previously accepted annotations in
/// the same project. <b>The similarity score is 1/(1 + d) of the nearest exemplar of the winning class in
/// standardised units — an uncalibrated similarity, not a probability.</b> A suggestion never changes
/// class or verification status by itself.
/// </summary>
public sealed class ClassSuggester
{
    public int K { get; init; } = 5;
    /// <summary>Only use exemplars at or above this verification level (default: any accepted annotation).</summary>
    public VerificationStatus MinimumExemplarVerification { get; init; } = VerificationStatus.Unverified;

    public ClassSuggestionResult? Suggest(AnnotationProject project, double[] descriptor, Guid? excludeAnnotation = null)
    {
        var exemplars = project.Images.SelectMany(i => i.Annotations)
            .Where(a => a.Descriptor is { Length: > 0 } && a.Id != excludeAnnotation && a.ClassId != Taxonomy.IgnoreId
                        && a.VerificationStatus >= MinimumExemplarVerification && a.ClassAssignmentSource == ClassAssignmentSource.Manual)
            .ToList();
        if (exemplars.Count == 0) return null;
        int dim = descriptor.Length;
        var mean = new double[dim]; var std = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            mean[d] = exemplars.Average(e => e.Descriptor![d]);
            double var = exemplars.Average(e => Math.Pow(e.Descriptor![d] - mean[d], 2));
            std[d] = Math.Sqrt(var) < 1e-9 ? 1 : Math.Sqrt(var);
        }
        double Dist(double[] x) => Math.Sqrt(Enumerable.Range(0, dim).Sum(d => Math.Pow((x[d] - descriptor[d]) / std[d], 2)));
        var nearest = exemplars.Select(e => (e, d: Dist(e.Descriptor!))).OrderBy(t => t.d).Take(K).ToList();
        var votes = nearest.GroupBy(t => t.e.ClassId)
            .Select(g => (cls: g.Key, w: g.Sum(t => 1.0 / (1e-6 + t.d)), nearest: g.Min(t => t.d), count: g.Count()))
            .OrderByDescending(v => v.w).ToList();
        var best = votes[0];
        double share = best.w / votes.Sum(v => v.w);
        var name = project.Taxonomy.Find(best.cls)?.Name ?? best.cls.ToString();
        return new ClassSuggestionResult(best.cls, name, 1.0 / (1.0 + best.nearest), share, nearest.Count,
            $"CIELAB colour and texture similarity to {best.count} of {nearest.Count} nearest previously annotated region(s) in this project");
    }

    /// <summary>Computes and stores the descriptor of an annotation (call after acceptance).</summary>
    public static double[] ComputeDescriptor(Mat image, Annotation a)
    {
        a.Descriptor = RegionStatistics.Compute(image, a.Mask).ToDescriptor();
        return a.Descriptor;
    }
}

public sealed record RegionCandidate(RasterMask Mask, int SuggestedClassId, double MeanDeltaE, string Reason);

/// <summary>
/// Smart region suggestions: finds unlabeled connected regions whose CIELAB colour is within ΔE00 ≤ T of
/// the median colour of an existing class in this image. Works on a downsampled copy (max 1 MP) for speed
/// and upsamples candidates with nearest-neighbour; candidates always require human acceptance.
/// </summary>
public static class RegionSuggester
{
    /// <summary>Median CIELAB colour per class from the annotations of an image (used for batch assistance).</summary>
    public static Dictionary<int, LabColor> ClassReferences(Mat image, ImageItem item)
    {
        var refs = new Dictionary<int, LabColor>();
        foreach (var g in item.Annotations.Where(a => a.ClassId != Taxonomy.IgnoreId && !a.Mask.IsEmpty).GroupBy(a => a.ClassId))
        {
            var mask = g.Select(a => a.Mask).Aggregate((x, y) => x.Union(y));
            var lab = LabImage.FromBgr(image, mask.Bounds);
            if (lab.Median(mask.Data) is { } m) refs[g.Key] = m;
        }
        return refs;
    }

    /// <param name="references">Class reference colours; when null they are taken from this image's own annotations.
    /// Passing references from another image implements batch-assisted annotation: every candidate still needs
    /// individual acceptance and verification on its own image.</param>
    public static List<RegionCandidate> Suggest(Mat image, ImageItem item, double deltaE = 8, int minArea = 50, int maxCandidates = 50,
        IReadOnlyDictionary<int, LabColor>? references = null, CancellationToken ct = default)
    {
        var result = new List<RegionCandidate>();
        var anns = item.Annotations.Where(a => a.ClassId != Taxonomy.IgnoreId && !a.Mask.IsEmpty).ToList();
        if (anns.Count == 0 && references is null) return result;
        double scale = Math.Min(1.0, Math.Sqrt(1_000_000.0 / ((double)image.Width * image.Height)));
        int w = Math.Max(1, (int)(image.Width * scale)), h = Math.Max(1, (int)(image.Height * scale));
        using var small = new Mat();
        Cv2.Resize(image, small, new Size(w, h), 0, 0, InterpolationFlags.Area);
        var lab = LabImage.FromBgr(small, new PixelRect(0, 0, w, h));

        var full = Export.LabelMapComposer.Compose(item);
        var labeledSmall = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                labeledSmall[y * w + x] = full.Labels[Math.Min(item.Height - 1, (int)(y / scale)) * item.Width + Math.Min(item.Width - 1, (int)(x / scale))];

        var classes = references?.Keys.ToList() ?? anns.Select(a => a.ClassId).Distinct().ToList();
        foreach (var cls in classes)
        {
            ct.ThrowIfCancellationRequested();
            LabColor reference;
            if (references is not null) reference = references[cls];
            else
            {
                var sel = new byte[w * h];
                for (int i = 0; i < sel.Length; i++) sel[i] = labeledSmall[i] == cls ? (byte)1 : (byte)0;
                if (lab.Median(sel) is not { } med) continue;
                reference = med;
            }
            var d = lab.DistanceMap(reference, ColorDifferenceMetric.CIEDE2000);
            var cand = new byte[w * h];
            for (int i = 0; i < cand.Length; i++) cand[i] = labeledSmall[i] == 0 && d[i] <= deltaE ? (byte)1 : (byte)0;
            using var m = MatInterop.FromBytes(cand, w, h);
            using var k = Vision.Morphology.MaskPostProcessor.Disk(1);
            Cv2.MorphologyEx(m, m, MorphTypes.Open, k);
            using var labels = new Mat(); using var stats = new Mat(); using var cents = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cents, PixelConnectivity.Connectivity8, MatType.CV_32S);
            labels.GetArray(out int[] lab32);
            for (int c = 1; c < n; c++)
            {
                double areaFull = stats.At<int>(c, (int)ConnectedComponentsTypes.Area) / (scale * scale);
                if (areaFull < minArea) continue;
                // Upsample this component to full resolution (nearest neighbour) and keep only unlabeled pixels.
                var bx = stats.At<int>(c, 0); var by = stats.At<int>(c, 1); var bw = stats.At<int>(c, 2); var bh = stats.At<int>(c, 3);
                var fb = new PixelRect((int)(bx / scale), (int)(by / scale), (int)Math.Ceiling(bw / scale) + 1, (int)Math.Ceiling(bh / scale) + 1).ClampTo(item.Width, item.Height);
                var data = new byte[fb.Width * fb.Height];
                double sumD = 0; int cnt = 0;
                for (int y = 0; y < fb.Height; y++)
                    for (int x = 0; x < fb.Width; x++)
                    {
                        int sx = Math.Min(w - 1, (int)((fb.X + x) * scale)), sy = Math.Min(h - 1, (int)((fb.Y + y) * scale));
                        if (lab32[sy * w + sx] != c || full.Labels[(fb.Y + y) * item.Width + fb.X + x] != 0) continue;
                        data[y * fb.Width + x] = 1; sumD += d[sy * w + sx]; cnt++;
                    }
                var mask = new RasterMask(fb, data).Trim();
                if (mask.IsEmpty) continue;
                result.Add(new RegionCandidate(mask, cls, sumD / Math.Max(1, cnt),
                    $"Unlabeled region with ΔE00 ≈ {sumD / Math.Max(1, cnt):0.0} to the reference colour of class {cls}" + (references is null ? " in this image" : " (batch reference)")));
            }
        }
        return result.OrderBy(r => r.MeanDeltaE).Take(maxCandidates).ToList();
    }
}

/// <summary>CSV export of per-annotation region statistics (section 19.1).</summary>
public static class RegionStatisticsCsv
{
    public static string Build(AnnotationProject project, ImageItem image, Mat source)
    {
        var sb = new StringBuilder();
        sb.AppendLine("image,annotation_id,class_id,class_name,verification,area_px,area_um2,perimeter_px,circularity,compactness,aspect_ratio," +
                      "mean_r,mean_g,mean_b,median_r,median_g,median_b,mean_L,mean_a,mean_b_star,std_L,std_a,std_b,min_L,max_L,min_a,max_a,min_b,max_b,edge_density,mean_gradient");
        foreach (var a in image.Annotations)
        {
            var s = RegionStatistics.Compute(source, a.Mask);
            string F(double v) => v.ToString("0.####", CultureInfo.InvariantCulture);
            var um2 = image.MicrometersPerPixel is { } mpp ? F(s.Area * mpp * mpp) : "";
            sb.AppendLine(string.Join(',', new[]
            {
                image.FileName, a.Id.ToString(), a.ClassId.ToString(), project.Taxonomy.Find(a.ClassId)?.Name ?? "", a.VerificationStatus.ToString(),
                s.Area.ToString(), um2, F(s.Perimeter), F(s.Circularity), F(s.Compactness), F(s.AspectRatio),
                F(s.MeanRgb[0]), F(s.MeanRgb[1]), F(s.MeanRgb[2]), F(s.MedianRgb[0]), F(s.MedianRgb[1]), F(s.MedianRgb[2]),
                F(s.MeanLab[0]), F(s.MeanLab[1]), F(s.MeanLab[2]), F(s.StdLab[0]), F(s.StdLab[1]), F(s.StdLab[2]),
                F(s.MinLab[0]), F(s.MaxLab[0]), F(s.MinLab[1]), F(s.MaxLab[1]), F(s.MinLab[2]), F(s.MaxLab[2]), F(s.EdgeDensity), F(s.MeanGradient),
            }));
        }
        return sb.ToString();
    }
}
