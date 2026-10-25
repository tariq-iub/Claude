namespace CuAnnotator.Vision.ColorAnalysis;

/// <summary>CIE L*a*b* colour (D65 reference white, L* in [0,100]).</summary>
public readonly record struct LabColor(double L, double A, double B)
{
    public override string ToString() => $"L*={L:0.00} a*={A:0.00} b*={B:0.00}";
}

/// <summary>HSV with H in degrees [0,360), S and V in [0,1].</summary>
public readonly record struct HsvColor(double H, double S, double V);

public enum ColorDifferenceMetric { CIEDE2000, CIE76 }

/// <summary>
/// Reference colour-science routines in double precision.
/// <para>
/// Assumptions: input RGB is 8-bit sRGB (IEC 61966-2-1) unless an ICC profile is applied upstream;
/// conversion to XYZ uses the sRGB/D65 matrix and Lab uses the D65 white point
/// (Xn = 0.950456, Yn = 1, Zn = 1.088754, the same constants OpenCV uses).
/// </para>
/// <para>
/// OpenCV representations (documented in 06_CIELAB_Color_Processing.md):
/// 8-bit <c>COLOR_BGR2Lab</c> stores L = L*·255/100, a = a*+128, b = b*+128;
/// 32-bit float input scaled to [0,1] yields L* ∈ [0,100] and unscaled a*, b*.
/// CuAnnotator always performs quantitative work on the float representation.
/// </para>
/// </summary>
public static class ColorScience
{
    public const double Xn = 0.950456, Yn = 1.0, Zn = 1.088754;

    public static double SrgbToLinear(double c) => c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);

    public static LabColor RgbToLab(byte r, byte g, byte b) => RgbToLab(r / 255.0, g / 255.0, b / 255.0);

    /// <summary>sRGB components in [0,1] → CIELAB (D65).</summary>
    public static LabColor RgbToLab(double r, double g, double b)
    {
        double rl = SrgbToLinear(r), gl = SrgbToLinear(g), bl = SrgbToLinear(b);
        double x = 0.412453 * rl + 0.357580 * gl + 0.180423 * bl;
        double y = 0.212671 * rl + 0.715160 * gl + 0.072169 * bl;
        double z = 0.019334 * rl + 0.119193 * gl + 0.950227 * bl;
        double fx = F(x / Xn), fy = F(y / Yn), fz = F(z / Zn);
        double L = y / Yn > 0.008856 ? 116 * fy - 16 : 903.3 * (y / Yn);
        return new LabColor(L, 500 * (fx - fy), 200 * (fy - fz));
    }

    private static double F(double t) => t > 0.008856 ? Math.Cbrt(t) : 7.787 * t + 16.0 / 116;

    /// <summary>Converts an OpenCV 8-bit Lab triple to CIELAB units.</summary>
    public static LabColor FromOpenCv8Bit(byte l, byte a, byte b) => new(l * 100.0 / 255.0, a - 128.0, b - 128.0);

    public static HsvColor RgbToHsv(byte r, byte g, byte b)
    {
        double rf = r / 255.0, gf = g / 255.0, bf = b / 255.0;
        double max = Math.Max(rf, Math.Max(gf, bf)), min = Math.Min(rf, Math.Min(gf, bf));
        double d = max - min, h = 0;
        if (d > 0)
        {
            if (max == rf) h = 60 * (((gf - bf) / d) % 6);
            else if (max == gf) h = 60 * ((bf - rf) / d + 2);
            else h = 60 * ((rf - gf) / d + 4);
        }
        if (h < 0) h += 360;
        return new HsvColor(h, max == 0 ? 0 : d / max, max);
    }

    public static double Gray(byte r, byte g, byte b) => 0.299 * r + 0.587 * g + 0.114 * b;

    public static double DeltaE76(LabColor p, LabColor q) =>
        Math.Sqrt(Sq(p.L - q.L) + Sq(p.A - q.A) + Sq(p.B - q.B));

    public static double Difference(LabColor p, LabColor q, ColorDifferenceMetric metric) =>
        metric == ColorDifferenceMetric.CIE76 ? DeltaE76(p, q) : DeltaE2000(p, q);

    /// <summary>
    /// CIEDE2000 colour difference with kL = kC = kH = 1, implemented as in
    /// Sharma, Wu &amp; Dalal (2005), Color Res. Appl. 30(1):21–30, including the hue-angle
    /// discontinuity handling; verified against the paper's 34 reference pairs in the test suite.
    /// </summary>
    public static double DeltaE2000(LabColor lab1, LabColor lab2, double kL = 1, double kC = 1, double kH = 1)
    {
        double L1 = lab1.L, a1 = lab1.A, b1 = lab1.B, L2 = lab2.L, a2 = lab2.A, b2 = lab2.B;
        double C1 = Math.Sqrt(a1 * a1 + b1 * b1), C2 = Math.Sqrt(a2 * a2 + b2 * b2);
        double Cbar = (C1 + C2) / 2;
        double Cbar7 = Math.Pow(Cbar, 7);
        double G = 0.5 * (1 - Math.Sqrt(Cbar7 / (Cbar7 + 6103515625.0))); // 25^7
        double a1p = (1 + G) * a1, a2p = (1 + G) * a2;
        double C1p = Math.Sqrt(a1p * a1p + b1 * b1), C2p = Math.Sqrt(a2p * a2p + b2 * b2);
        double h1p = HueDeg(b1, a1p), h2p = HueDeg(b2, a2p);

        double dLp = L2 - L1, dCp = C2p - C1p;
        double dhp;
        if (C1p * C2p == 0) dhp = 0;
        else
        {
            dhp = h2p - h1p;
            if (dhp > 180) dhp -= 360;
            else if (dhp < -180) dhp += 360;
        }
        double dHp = 2 * Math.Sqrt(C1p * C2p) * Math.Sin(Rad(dhp / 2));

        double Lbarp = (L1 + L2) / 2, Cbarp = (C1p + C2p) / 2;
        double hbarp;
        if (C1p * C2p == 0) hbarp = h1p + h2p;
        else if (Math.Abs(h1p - h2p) <= 180) hbarp = (h1p + h2p) / 2;
        else if (h1p + h2p < 360) hbarp = (h1p + h2p + 360) / 2;
        else hbarp = (h1p + h2p - 360) / 2;

        double T = 1 - 0.17 * Math.Cos(Rad(hbarp - 30)) + 0.24 * Math.Cos(Rad(2 * hbarp))
                 + 0.32 * Math.Cos(Rad(3 * hbarp + 6)) - 0.20 * Math.Cos(Rad(4 * hbarp - 63));
        double dTheta = 30 * Math.Exp(-Sq((hbarp - 275) / 25));
        double Cbarp7 = Math.Pow(Cbarp, 7);
        double RC = 2 * Math.Sqrt(Cbarp7 / (Cbarp7 + 6103515625.0));
        double SL = 1 + 0.015 * Sq(Lbarp - 50) / Math.Sqrt(20 + Sq(Lbarp - 50));
        double SC = 1 + 0.045 * Cbarp;
        double SH = 1 + 0.015 * Cbarp * T;
        double RT = -Math.Sin(Rad(2 * dTheta)) * RC;

        double tl = dLp / (kL * SL), tc = dCp / (kC * SC), th = dHp / (kH * SH);
        return Math.Sqrt(tl * tl + tc * tc + th * th + RT * tc * th);
    }

    private static double HueDeg(double b, double ap)
    {
        if (b == 0 && ap == 0) return 0;
        double h = Math.Atan2(b, ap) * 180 / Math.PI;
        return h < 0 ? h + 360 : h;
    }

    private static double Rad(double deg) => deg * Math.PI / 180;
    private static double Sq(double v) => v * v;
}
