using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Desktop.Controls;

/// <summary>Navigator: thumbnail of the whole image with the visible region; click or drag to navigate.</summary>
public sealed class Minimap : FrameworkElement
{
    private ImageViewport? _viewport;
    private BitmapSource? _thumb;
    private int _imgW, _imgH;

    public void Attach(ImageViewport viewport)
    {
        _viewport = viewport;
        viewport.TransformChanged += (_, _) => InvalidateVisual();
        viewport.SizeChanged += (_, _) => InvalidateVisual();
    }

    public void SetImage(BitmapSource? thumb, int width, int height)
    {
        _thumb = thumb; _imgW = width; _imgH = height;
        InvalidateVisual();
    }

    private Rect ThumbRect()
    {
        if (_thumb is null || _imgW == 0) return Rect.Empty;
        double s = Math.Min(ActualWidth / _imgW, ActualHeight / _imgH);
        double w = _imgW * s, h = _imgH * s;
        return new Rect((ActualWidth - w) / 2, (ActualHeight - h) / 2, w, h);
    }

    protected override void OnRender(DrawingContext dc)
    {
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
        var r = ThumbRect();
        if (_thumb is null || r.IsEmpty || _viewport is null) return;
        dc.DrawImage(_thumb, r);
        var v = _viewport.VisibleImageRect();
        double s = r.Width / _imgW;
        var vr = new Rect(r.X + v.X * s, r.Y + v.Y * s, v.Width * s, v.Height * s);
        vr.Intersect(r);
        if (!vr.IsEmpty) dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(40, 47, 111, 208)), new Pen(Brushes.DodgerBlue, 1.5), vr);
    }

    protected override void OnMouseDown(MouseButtonEventArgs e) { Navigate(e.GetPosition(this)); CaptureMouse(); }
    protected override void OnMouseMove(MouseEventArgs e) { if (IsMouseCaptured) Navigate(e.GetPosition(this)); }
    protected override void OnMouseUp(MouseButtonEventArgs e) => ReleaseMouseCapture();

    private void Navigate(Point p)
    {
        var r = ThumbRect();
        if (r.IsEmpty || _viewport is null) return;
        double s = r.Width / _imgW;
        _viewport.CenterOn(new ImagePoint((p.X - r.X) / s, (p.Y - r.Y) / s));
    }
}
