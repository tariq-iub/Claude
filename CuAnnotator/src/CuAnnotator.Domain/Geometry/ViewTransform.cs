namespace CuAnnotator.Domain.Geometry;

/// <summary>
/// Maps continuous image coordinates to viewport coordinates (WPF device-independent pixels, DIPs):
/// <c>view = image × Scale + Offset</c>.
/// <para>
/// WPF applies the monitor DPI scale (DIP → physical pixel) after this transform, so annotations stored
/// in image space are independent of zoom, pan, window size and DPI. All conversions are pure and
/// unit-tested; stored geometry is never derived from screen pixels without passing through
/// <see cref="ViewToImage"/>.
/// </para>
/// </summary>
public sealed record ViewTransform(double Scale, double OffsetX, double OffsetY)
{
    public const double MinScale = 1.0 / 64;
    public const double MaxScale = 64;

    public static ViewTransform Identity { get; } = new(1, 0, 0);

    public ImagePoint ViewToImage(double vx, double vy) => new((vx - OffsetX) / Scale, (vy - OffsetY) / Scale);

    public (double X, double Y) ImageToView(ImagePoint p) => (p.X * Scale + OffsetX, p.Y * Scale + OffsetY);

    /// <summary>Pixel under a viewport location (floor rule, may lie outside the image).</summary>
    public PixelPoint ViewToPixel(double vx, double vy) => ViewToImage(vx, vy).ToPixel();

    /// <summary>Zoom by <paramref name="factor"/> keeping the image point under (vx, vy) fixed.</summary>
    public ViewTransform ZoomAt(double factor, double vx, double vy)
    {
        double newScale = Math.Clamp(Scale * factor, MinScale, MaxScale);
        var anchor = ViewToImage(vx, vy);
        return new ViewTransform(newScale, vx - anchor.X * newScale, vy - anchor.Y * newScale);
    }

    public ViewTransform WithScaleAt(double scale, double vx, double vy) => ZoomAt(scale / Scale, vx, vy);

    public ViewTransform Pan(double dx, double dy) => this with { OffsetX = OffsetX + dx, OffsetY = OffsetY + dy };

    /// <summary>Largest scale that shows the whole image, centred in the viewport.</summary>
    public static ViewTransform Fit(int imageWidth, int imageHeight, double viewWidth, double viewHeight, double margin = 8)
    {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 2 * margin || viewHeight <= 2 * margin)
            return Identity;
        double s = Math.Clamp(Math.Min((viewWidth - 2 * margin) / imageWidth, (viewHeight - 2 * margin) / imageHeight), MinScale, MaxScale);
        return new ViewTransform(s, (viewWidth - imageWidth * s) / 2, (viewHeight - imageHeight * s) / 2);
    }

    /// <summary>
    /// 100 % = one image pixel per physical screen pixel. <paramref name="dpiScale"/> is the WPF DPI
    /// factor (1.0 at 96 DPI, 1.5 at 144 DPI), so the DIP scale is 1/dpiScale.
    /// </summary>
    public static ViewTransform ActualPixels(double dpiScale, double vx, double vy, ViewTransform current) =>
        current.WithScaleAt(1.0 / dpiScale, vx, vy);

    /// <summary>Visible image region (continuous) for a viewport of the given size.</summary>
    public (ImagePoint TopLeft, ImagePoint BottomRight) VisibleImageRegion(double viewWidth, double viewHeight) =>
        (ViewToImage(0, 0), ViewToImage(viewWidth, viewHeight));
}
