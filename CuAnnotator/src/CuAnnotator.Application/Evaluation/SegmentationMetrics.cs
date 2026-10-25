namespace CuAnnotator.Application.Evaluation;

public sealed record BinaryMetrics(long TP, long FP, long FN, long TN, double IoU, double Dice, double Precision, double Recall, double PixelAccuracy);

public sealed record AgreementMetrics(double PixelAgreement, double CohenKappa, double MeanIoU, IReadOnlyDictionary<int, double> PerClassIoU, IReadOnlyDictionary<int, double> PerClassDice);

/// <summary>
/// Evaluation metrics over full-frame label arrays (row-major). Conventions: empty ∩ empty gives IoU = Dice = 1;
/// precision/recall with an empty denominator are reported as NaN. Pixels with value
/// <paramref name="ignoreValue"/> in the reference are excluded.
/// </summary>
public static class SegmentationMetrics
{
    public static BinaryMetrics Binary(ReadOnlySpan<byte> prediction, ReadOnlySpan<byte> reference, ReadOnlySpan<byte> ignore = default)
    {
        if (prediction.Length != reference.Length) throw new ArgumentException("Size mismatch.");
        long tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            if (!ignore.IsEmpty && ignore[i] != 0) continue;
            bool p = prediction[i] != 0, r = reference[i] != 0;
            if (p && r) tp++; else if (p) fp++; else if (r) fn++; else tn++;
        }
        double iou = tp + fp + fn == 0 ? 1 : (double)tp / (tp + fp + fn);
        double dice = 2 * tp + fp + fn == 0 ? 1 : 2.0 * tp / (2 * tp + fp + fn);
        double prec = tp + fp == 0 ? double.NaN : (double)tp / (tp + fp);
        double rec = tp + fn == 0 ? double.NaN : (double)tp / (tp + fn);
        double acc = (double)(tp + tn) / Math.Max(1, tp + tn + fp + fn);
        return new BinaryMetrics(tp, fp, fn, tn, iou, dice, prec, rec, acc);
    }

    /// <summary>
    /// Boundary F1 (Csurka et al., 2013 BF score): boundary pixels of prediction P and reference R
    /// (foreground pixels with a 4-neighbour in background or outside the frame) are matched within a
    /// Euclidean tolerance <paramref name="tolerance"/> px. Precision = fraction of P-boundary within
    /// tolerance of R-boundary; recall symmetric; F1 = harmonic mean.
    /// </summary>
    public static (double Precision, double Recall, double F1) BoundaryF1(ReadOnlySpan<byte> prediction, ReadOnlySpan<byte> reference, int width, int height, double tolerance = 2.0)
    {
        var bp = BoundaryPoints(prediction, width, height);
        var br = BoundaryPoints(reference, width, height);
        if (bp.Count == 0 && br.Count == 0) return (1, 1, 1);
        if (bp.Count == 0 || br.Count == 0) return (0, 0, 0);
        var dtR = DistanceTransform(br, width, height);
        var dtP = DistanceTransform(bp, width, height);
        double precision = bp.Count(p => dtR[p.Y * width + p.X] <= tolerance) / (double)bp.Count;
        double recall = br.Count(p => dtP[p.Y * width + p.X] <= tolerance) / (double)br.Count;
        double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return (precision, recall, f1);
    }

    private static List<(int X, int Y)> BoundaryPoints(ReadOnlySpan<byte> m, int w, int h)
    {
        var pts = new List<(int, int)>();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                if (m[i] == 0) continue;
                if (x == 0 || y == 0 || x == w - 1 || y == h - 1 || m[i - 1] == 0 || m[i + 1] == 0 || m[i - w] == 0 || m[i + w] == 0)
                    pts.Add((x, y));
            }
        return pts;
    }

    /// <summary>Exact Euclidean distance transform to the given points (Felzenszwalb–Huttenlocher, separable).</summary>
    private static double[] DistanceTransform(List<(int X, int Y)> pts, int w, int h)
    {
        const double Inf = 1e20;
        var f = new double[w * h];
        Array.Fill(f, Inf);
        foreach (var (x, y) in pts) f[y * w + x] = 0;
        var tmp = new double[Math.Max(w, h)];
        var res = new double[Math.Max(w, h)];
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++) tmp[y] = f[y * w + x];
            Dt1(tmp, h, res);
            for (int y = 0; y < h; y++) f[y * w + x] = res[y];
        }
        for (int y = 0; y < h; y++)
        {
            Array.Copy(f, y * w, tmp, 0, w);
            Dt1(tmp, w, res);
            for (int x = 0; x < w; x++) f[y * w + x] = Math.Sqrt(res[x]);
        }
        return f;
    }

    private static void Dt1(double[] f, int n, double[] d)
    {
        var v = new int[n]; var z = new double[n + 1];
        int k = 0; v[0] = 0; z[0] = double.NegativeInfinity; z[1] = double.PositiveInfinity;
        for (int q = 1; q < n; q++)
        {
            double s;
            while (true)
            {
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k]);
                if (s <= z[k] && k > 0) k--; else break;
            }
            if (s <= z[k]) { v[0] = q; z[0] = double.NegativeInfinity; z[1] = double.PositiveInfinity; k = 0; continue; }
            k++; v[k] = q; z[k] = s; z[k + 1] = double.PositiveInfinity;
        }
        k = 0;
        for (int q = 0; q < n; q++)
        {
            while (z[k + 1] < q) k++;
            d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
        }
    }

    /// <summary>
    /// Multiclass agreement between two label maps (e.g. two annotators). Pixels where either map is
    /// <paramref name="ignoreValue"/> are excluded; by default pixels unlabeled (0) in both maps are excluded too.
    /// Cohen's κ = (p_o − p_e) / (1 − p_e).
    /// </summary>
    public static AgreementMetrics Agreement(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, byte ignoreValue = 255, bool excludeJointUnlabeled = true)
    {
        if (a.Length != b.Length) throw new ArgumentException("Size mismatch.");
        var conf = new long[256, 256];
        long n = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] == ignoreValue || b[i] == ignoreValue) continue;
            if (excludeJointUnlabeled && a[i] == 0 && b[i] == 0) continue;
            conf[a[i], b[i]]++; n++;
        }
        if (n == 0) return new AgreementMetrics(1, 1, 1, new Dictionary<int, double>(), new Dictionary<int, double>());
        long agree = 0;
        var rowSum = new long[256]; var colSum = new long[256];
        for (int i = 0; i < 256; i++)
            for (int j = 0; j < 256; j++) { rowSum[i] += conf[i, j]; colSum[j] += conf[i, j]; if (i == j) agree += conf[i, j]; }
        double po = (double)agree / n;
        double pe = 0;
        for (int c = 0; c < 256; c++) pe += (double)rowSum[c] * colSum[c] / ((double)n * n);
        double kappa = pe >= 1 ? 1 : (po - pe) / (1 - pe);
        var iou = new Dictionary<int, double>(); var dice = new Dictionary<int, double>();
        for (int c = 1; c < 256; c++)
        {
            if (rowSum[c] + colSum[c] == 0) continue;
            long tp = conf[c, c];
            iou[c] = (double)tp / (rowSum[c] + colSum[c] - tp);
            dice[c] = 2.0 * tp / (rowSum[c] + colSum[c]);
        }
        return new AgreementMetrics(po, kappa, iou.Count == 0 ? 1 : iou.Values.Average(), iou, dice);
    }
}
