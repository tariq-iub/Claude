using System.Globalization;
using System.Text;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

namespace CuAnnotator.Application.Evaluation;

public sealed record AlgorithmEvaluation(string Algorithm, BinaryMetrics Metrics, double BoundaryF1, double Milliseconds, long PredictedPixels);

/// <summary>Research evaluation: algorithm comparison against reference masks and annotation-efficiency reporting.</summary>
public static class EvaluationService
{
    /// <summary>
    /// Runs each algorithm on the same ROI and compares its result with a reference mask
    /// (inside the ROI only). Boundary F1 uses <paramref name="boundaryTolerance"/> px.
    /// </summary>
    public static async Task<List<AlgorithmEvaluation>> CompareAlgorithmsAsync(SegmentationEngine engine, Mat image, PixelRect roi,
        RasterMask reference, SegmentationParameters baseParameters, IEnumerable<SegmentationAlgorithmKind> kinds,
        double boundaryTolerance = 2.0, CancellationToken ct = default)
    {
        var results = new List<AlgorithmEvaluation>();
        var frame = roi.ClampTo(image.Width, image.Height);
        var refLocal = reference.Reframe(frame).Data;
        foreach (var k in kinds)
        {
            var p = baseParameters.Clone();
            p.Algorithm = k;
            var r = await engine.SegmentAsync(new SegmentationRequest { Image = image, Roi = frame, Parameters = p }, ct);
            var pred = r.Mask.Reframe(frame).Data;
            var m = SegmentationMetrics.Binary(pred, refLocal);
            var bf = SegmentationMetrics.BoundaryF1(pred, refLocal, frame.Width, frame.Height, boundaryTolerance);
            results.Add(new AlgorithmEvaluation(r.AlgorithmName, m, bf.F1, r.ProcessingTime.TotalMilliseconds, r.Mask.PixelCount));
        }
        return results;
    }

    public static string ComparisonCsv(IEnumerable<AlgorithmEvaluation> rows, double tolerance)
    {
        var sb = new StringBuilder($"algorithm,iou,dice,precision,recall,pixel_accuracy,boundary_f1_tol{tolerance.ToString(CultureInfo.InvariantCulture)}px,ms,predicted_pixels\n");
        foreach (var r in rows)
            sb.AppendLine(string.Join(',', QualityReport.Csv(r.Algorithm), F(r.Metrics.IoU), F(r.Metrics.Dice), F(r.Metrics.Precision),
                F(r.Metrics.Recall), F(r.Metrics.PixelAccuracy), F(r.BoundaryF1), F(r.Milliseconds), r.PredictedPixels));
        return sb.ToString();
    }

    /// <summary>
    /// Per-annotation efficiency log recorded at acceptance: interaction count (ROI + strokes), refinement runs,
    /// seconds from ROI to acceptance, fraction of the first automatic proposal retained.
    /// </summary>
    public static string EfficiencyCsv(AnnotationProject project)
    {
        var sb = new StringBuilder("image,annotation_id,class_id,algorithm,interactions,refinement_runs,seconds,first_proposal_retained,area_px,geometry_source,revision\n");
        foreach (var img in project.Images)
            foreach (var a in img.Annotations)
            {
                string P(string k) => a.AlgorithmParameters.TryGetValue(k, out var v) ? v : "";
                sb.AppendLine(string.Join(',', QualityReport.Csv(img.FileName), a.Id, a.ClassId, QualityReport.Csv(a.SegmentationAlgorithm),
                    P("efficiency.interactions"), P("efficiency.refinementRuns"), P("efficiency.seconds"), P("efficiency.firstProposalRetained"),
                    a.PixelArea, a.GeometrySource, a.Revision));
            }
        return sb.ToString();
    }

    public sealed record EfficiencySummary(int Annotations, double TotalSeconds, double MeanSecondsPerRegion, double MeanInteractions, double MeanRefinementRuns, double MeanFirstProposalRetained, int Segmented, int ManualOrEdited);

    public static EfficiencySummary Summarize(AnnotationProject project)
    {
        var anns = project.Images.SelectMany(i => i.Annotations).ToList();
        double D(Domain.Entities.Annotation a, string k) =>
            a.AlgorithmParameters.TryGetValue(k, out var v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : double.NaN;
        IEnumerable<double> Vals(string k) => anns.Select(a => D(a, k)).Where(d => !double.IsNaN(d));
        double Mean(string k) => Vals(k).DefaultIfEmpty(double.NaN).Average();
        return new EfficiencySummary(anns.Count, Vals("efficiency.seconds").Sum(), Mean("efficiency.seconds"), Mean("efficiency.interactions"),
            Mean("efficiency.refinementRuns"), Mean("efficiency.firstProposalRetained"),
            anns.Count(a => a.GeometrySource == GeometrySource.Segmentation), anns.Count(a => a.GeometrySource != GeometrySource.Segmentation));
    }

    /// <summary>Inter-annotator agreement between two annotation sets on one image.</summary>
    public static (AgreementMetrics Agreement, double BoundaryF1) InterAnnotator(ImageItem image, string setA, string setB, double tolerance = 2.0)
    {
        var a = Export.LabelMapComposer.Compose(image, setA).Labels;
        var b = Export.LabelMapComposer.Compose(image, setB).Labels;
        var agreement = SegmentationMetrics.Agreement(a, b);
        var fa = a.Select(v => v is not 0 and not 255 ? (byte)1 : (byte)0).ToArray();
        var fb = b.Select(v => v is not 0 and not 255 ? (byte)1 : (byte)0).ToArray();
        return (agreement, SegmentationMetrics.BoundaryF1(fa, fb, image.Width, image.Height, tolerance).F1);
    }

    private static string F(double v) => double.IsNaN(v) ? "" : v.ToString("0.#####", CultureInfo.InvariantCulture);
}
