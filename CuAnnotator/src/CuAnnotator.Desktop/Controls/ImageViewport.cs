using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using CuAnnotator.Desktop.Interaction;
using CuAnnotator.Desktop.Rendering;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Desktop.Controls;

public sealed class ImagePointerEventArgs(ImagePoint image, Point view, MouseButton? button, ModifierKeys modifiers, int clickCount) : EventArgs
{
    public ImagePoint Image { get; } = image;
    public PixelPoint Pixel => Image.ToPixel();
    public Point View { get; } = view;
    public MouseButton? Button { get; } = button;
    public ModifierKeys Modifiers { get; } = modifiers;
    public int ClickCount { get; } = clickCount;
    public bool Handled { get; set; }
}

/// <summary>
/// Central image workspace. Draws the tiled image pyramid, annotation overlays, candidate segmentation,
/// tool previews and the pixel grid. Owns the <see cref="ViewTransform"/> (zoom/pan) and converts every
/// mouse position to image space with it before raising pointer events; nothing outside this class ever
/// sees screen coordinates. WPF's DIP→device scaling is applied after this transform, so DPI changes
/// never affect image-space geometry.
/// </summary>
public sealed class ImageViewport : FrameworkElement
{
    public static readonly DependencyProperty SceneProperty = DependencyProperty.Register(nameof(Scene), typeof(ViewportScene), typeof(ImageViewport),
        new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, (d, e) => ((ImageViewport)d).OnSceneChanged(e)));

    public ViewportScene? Scene { get => (ViewportScene?)GetValue(SceneProperty); set => SetValue(SceneProperty, value); }

    public ViewTransform Transform { get; private set; } = ViewTransform.Identity;

    /// <summary>When true, a left-drag pans (Select/Pan tool or Space held).</summary>
    public bool PanWithLeftButton { get; set; }
    public bool SpaceHeld { get; set; }

    public event EventHandler<ImagePointerEventArgs>? ImagePointerDown;
    public event EventHandler<ImagePointerEventArgs>? ImagePointerMove;
    public event EventHandler<ImagePointerEventArgs>? ImagePointerUp;
    public event EventHandler? TransformChanged;

    private Point? _panStart;
    private ViewTransform _panOrigin = ViewTransform.Identity;
    private bool _fitPending = true;

    private static readonly Brush CheckerA = Freeze(new SolidColorBrush(Color.FromRgb(0x2A, 0x2F, 0x37)));

    public ImageViewport()
    {
        Focusable = true;
        ClipToBounds = true;
        Cursor = Cursors.Cross;
        SizeChanged += (_, _) => { if (_fitPending) FitToWindow(); else InvalidateVisual(); };
    }

    private void OnSceneChanged(DependencyPropertyChangedEventArgs e)
    {
        if (e.OldValue is ViewportScene o) o.Changed -= OnSceneInvalidated;
        if (e.NewValue is ViewportScene n) n.Changed += OnSceneInvalidated;
        _fitPending = true;
        FitToWindow();
    }

    private void OnSceneInvalidated(object? sender, EventArgs e) => InvalidateVisual();

    // ───────────── Zoom & pan ─────────────

    private void SetTransform(ViewTransform t)
    {
        Transform = t;
        InvalidateVisual();
        TransformChanged?.Invoke(this, EventArgs.Empty);
    }

    public void FitToWindow()
    {
        if (Scene?.Image is not { } img || ActualWidth <= 0) return;
        _fitPending = false;
        SetTransform(ViewTransform.Fit(img.Width, img.Height, ActualWidth, ActualHeight));
    }

    /// <summary>100 %: one image pixel per physical screen pixel on the current monitor.</summary>
    public void ActualPixels()
    {
        _fitPending = false;
        double dpi = VisualTreeHelper.GetDpi(this).DpiScaleX;
        SetTransform(ViewTransform.ActualPixels(dpi, ActualWidth / 2, ActualHeight / 2, Transform));
    }

    public void ZoomBy(double factor, Point? at = null)
    {
        _fitPending = false;
        var p = at ?? new Point(ActualWidth / 2, ActualHeight / 2);
        SetTransform(Transform.ZoomAt(factor, p.X, p.Y));
    }

    /// <summary>Centres the view on an image location (minimap navigation, review queue).</summary>
    public void CenterOn(ImagePoint p)
    {
        _fitPending = false;
        var (vx, vy) = Transform.ImageToView(p);
        SetTransform(Transform.Pan(ActualWidth / 2 - vx, ActualHeight / 2 - vy));
    }

    /// <summary>Zoom percentage relative to physical pixels (100 % = 1 image px per device px).</summary>
    public double ZoomPercent => Transform.Scale * VisualTreeHelper.GetDpi(this).DpiScaleX * 100;

    public Rect VisibleImageRect()
    {
        var (a, b) = Transform.VisibleImageRegion(ActualWidth, ActualHeight);
        return new Rect(new Point(a.X, a.Y), new Point(b.X, b.Y));
    }

    // ───────────── Input ─────────────

    private ImagePointerEventArgs Args(MouseEventArgs e, MouseButton? button, int clicks = 1)
    {
        var v = e.GetPosition(this);
        return new ImagePointerEventArgs(Transform.ViewToImage(v.X, v.Y), v, button, Keyboard.Modifiers, clicks);
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        _fitPending = false;
        double factor = Math.Pow(1.0015, e.Delta); // ≈ ×1.2 per notch, smooth for high-resolution wheels
        var p = e.GetPosition(this);
        SetTransform(Transform.ZoomAt(factor, p.X, p.Y));
        e.Handled = true;
    }

    protected override void OnMouseDown(MouseButtonEventArgs e)
    {
        Focus();
        bool pan = e.ChangedButton == MouseButton.Middle || (e.ChangedButton == MouseButton.Left && (SpaceHeld || PanWithLeftButton));
        if (e.ChangedButton == MouseButton.Left && PanWithLeftButton && !SpaceHeld)
        {
            // Select/Pan tool: a click selects; a drag pans. Let the tool see the press first.
            var a = Args(e, e.ChangedButton, e.ClickCount);
            ImagePointerDown?.Invoke(this, a);
        }
        if (pan)
        {
            _panStart = e.GetPosition(this);
            _panOrigin = Transform;
            CaptureMouse();
            Cursor = Cursors.SizeAll;
            e.Handled = true;
            return;
        }
        var args = Args(e, e.ChangedButton, e.ClickCount);
        ImagePointerDown?.Invoke(this, args);
        CaptureMouse();
        e.Handled = true;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        if (_panStart is { } s)
        {
            var p = e.GetPosition(this);
            _fitPending = false;
            SetTransform(_panOrigin.Pan(p.X - s.X, p.Y - s.Y));
        }
        ImagePointerMove?.Invoke(this, Args(e, null));
    }

    protected override void OnMouseUp(MouseButtonEventArgs e)
    {
        if (_panStart is not null)
        {
            _panStart = null;
            ReleaseMouseCapture();
            Cursor = Cursors.Cross;
            if (!(e.ChangedButton == MouseButton.Left && PanWithLeftButton)) { e.Handled = true; return; }
        }
        ImagePointerUp?.Invoke(this, Args(e, e.ChangedButton));
        ReleaseMouseCapture();
        e.Handled = true;
    }

    // ───────────── Rendering ─────────────

    protected override void OnRender(DrawingContext dc)
    {
        var bg = (Brush?)TryFindResource("ViewportBackground") ?? Brushes.DimGray;
        dc.DrawRectangle(bg, null, new Rect(0, 0, ActualWidth, ActualHeight));
        var scene = Scene;
        if (scene?.Image is not { } img) return;

        var t = Transform;
        double s = t.Scale;
        var visible = VisibleImageRect();
        visible.Intersect(new Rect(0, 0, img.Width, img.Height));
        if (visible.IsEmpty) return;

        // Pixel-exact (nearest) magnification above 100 %, smooth minification below.
        RenderOptions.SetBitmapScalingMode(this, s >= 1 ? BitmapScalingMode.NearestNeighbor : BitmapScalingMode.HighQuality);
        dc.PushTransform(new MatrixTransform(s, 0, 0, s, t.OffsetX, t.OffsetY));

        dc.DrawRectangle(CheckerA, null, new Rect(0, 0, img.Width, img.Height));
        foreach (var (bmp, rect) in img.VisibleTiles(img.LevelFor(s), visible))
            dc.DrawImage(bmp, rect);

        bool overlays = scene.Mode != OverlayMode.Hidden;
        if (overlays)
        {
            Geometry? clip = OverlayClip(scene.Mode, t);
            if (clip is not null) dc.PushClip(clip);
            DrawAnnotations(dc, scene, s);
            if (clip is not null) dc.Pop();
        }
        DrawCandidate(dc, scene, s);
        foreach (var g in scene.Guidance) DrawShape(dc, g, s);
        foreach (var p in scene.Preview) DrawShape(dc, p, s);
        DrawEditPolygons(dc, scene, s);
        if (scene.ShowGrid && s >= 8) DrawGrid(dc, visible, s);

        dc.Pop();

        if (scene.Mode == OverlayMode.Split)
        {
            var pen = new Pen((Brush?)TryFindResource("Accent") ?? Brushes.DodgerBlue, 2);
            dc.DrawLine(pen, new Point(ActualWidth / 2, 0), new Point(ActualWidth / 2, ActualHeight));
        }
    }

    /// <summary>Clip geometry in image space for split-view and checkerboard comparison modes.</summary>
    private Geometry? OverlayClip(OverlayMode mode, ViewTransform t)
    {
        if (mode == OverlayMode.Split)
        {
            var a = t.ViewToImage(ActualWidth / 2, -10);
            var b = t.ViewToImage(ActualWidth + 10, ActualHeight + 10);
            return new RectangleGeometry(new Rect(new Point(a.X, a.Y), new Point(b.X, b.Y)));
        }
        if (mode == OverlayMode.Checkerboard)
        {
            const double cell = 48; // screen DIPs
            var group = new GeometryGroup();
            for (int cy = 0; cy * cell < ActualHeight; cy++)
                for (int cx = 0; cx * cell < ActualWidth; cx++)
                {
                    if ((cx + cy) % 2 == 1) continue;
                    var a = t.ViewToImage(cx * cell, cy * cell);
                    var b = t.ViewToImage((cx + 1) * cell, (cy + 1) * cell);
                    group.Children.Add(new RectangleGeometry(new Rect(new Point(a.X, a.Y), new Point(b.X, b.Y))));
                }
            return group;
        }
        return null;
    }

    private static void DrawAnnotations(DrawingContext dc, ViewportScene scene, double s)
    {
        bool contoursOnly = scene.Mode == OverlayMode.ContoursOnly;
        if (scene.ShowMasks && !contoursOnly)
        {
            dc.PushOpacity(scene.Opacity);
            foreach (var a in scene.Annotations)
                if (a.Visible && a.Fill is not null) dc.DrawImage(a.Fill, a.FillRect);
            dc.Pop();
        }
        if (scene.ShowBoundaries || contoursOnly)
            foreach (var a in scene.Annotations)
            {
                if (!a.Visible) continue;
                bool sel = scene.Selected.Contains(a.Id);
                if (contoursOnly) dc.DrawGeometry(null, new Pen(Brushes.Black, 3 / s), a.Outline);
                var pen = new Pen(new SolidColorBrush(contoursOnly ? Colors.Yellow : a.Color), (sel ? 2.5 : 1.2) / s);
                dc.DrawGeometry(null, pen, a.Outline);
                if (sel)
                {
                    var dash = new Pen(Brushes.White, 1.2 / s) { DashStyle = new DashStyle(new[] { 4.0, 4.0 }, 0) };
                    dc.DrawGeometry(null, dash, a.Outline);
                }
            }
    }

    private static void DrawCandidate(DrawingContext dc, ViewportScene scene, double s)
    {
        if (scene.CandidateRoi is { } roi)
        {
            var pen = new Pen(Brushes.DeepSkyBlue, 1.5 / s) { DashStyle = new DashStyle(new[] { 6.0, 4.0 }, 0) };
            dc.DrawRectangle(null, pen, new Rect(roi.X, roi.Y, roi.Width, roi.Height));
        }
        if (scene.Candidate is not { } c) return;
        if (c.Fill is not null) { dc.PushOpacity(0.45); dc.DrawImage(c.Fill, c.FillRect); dc.Pop(); }
        dc.DrawGeometry(null, new Pen(Brushes.Black, 3 / s), c.Outline);
        dc.DrawGeometry(null, new Pen(Brushes.Cyan, 1.5 / s), c.Outline);
    }

    private static void DrawShape(DrawingContext dc, PreviewShape p, double s)
    {
        if (p.Points.Count == 0) return;
        double th = p.ImageThickness > 0 ? p.ImageThickness : p.ScreenThickness / s;
        var brush = new SolidColorBrush(p.Color);
        var pen = new Pen(brush, th) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round, LineJoin = PenLineJoin.Round };
        if (p.Dashed) pen.DashStyle = new DashStyle(new[] { 4.0, 3.0 }, 0);
        if (p.Points.Count == 1)
        {
            dc.DrawEllipse(brush, null, new Point(p.Points[0].X, p.Points[0].Y), th / 2, th / 2);
            return;
        }
        var g = new StreamGeometry();
        using (var ctx = g.Open())
        {
            ctx.BeginFigure(new Point(p.Points[0].X, p.Points[0].Y), false, p.Closed);
            ctx.PolyLineTo(p.Points.Skip(1).Select(q => new Point(q.X, q.Y)).ToList(), true, true);
        }
        dc.DrawGeometry(null, pen, g);
    }

    private static void DrawEditPolygons(DrawingContext dc, ViewportScene scene, double s)
    {
        if (scene.EditPolygons is not { } polys) return;
        dc.DrawGeometry(null, new Pen(Brushes.Yellow, 1.5 / s), OverlayFactory.Outline(polys));
        double h = 4 / s;
        int total = polys.Sum(p => p.VertexCount);
        if (total > 4000) return; // handles would be unreadable; the status bar suggests simplifying
        for (int c = 0; c < polys.Count; c++)
        {
            var rings = polys[c].Holes.Prepend(polys[c].Outer).ToList();
            for (int r = 0; r < rings.Count; r++)
                for (int i = 0; i < rings[r].Count; i++)
                {
                    bool hot = scene.HotVertex == (c, r, i);
                    var p = rings[r][i];
                    dc.DrawRectangle(hot ? Brushes.OrangeRed : Brushes.White, new Pen(Brushes.Black, 1 / s),
                        new Rect(p.X + 0.5 - h, p.Y + 0.5 - h, 2 * h, 2 * h));
                }
        }
    }

    private static void DrawGrid(DrawingContext dc, Rect visible, double s)
    {
        var pen = new Pen(new SolidColorBrush(Color.FromArgb(70, 128, 128, 128)), 1 / s);
        pen.Freeze();
        int x0 = (int)Math.Floor(visible.Left), x1 = (int)Math.Ceiling(visible.Right);
        int y0 = (int)Math.Floor(visible.Top), y1 = (int)Math.Ceiling(visible.Bottom);
        for (int x = x0; x <= x1; x++) dc.DrawLine(pen, new Point(x, y0), new Point(x, y1));
        for (int y = y0; y <= y1; y++) dc.DrawLine(pen, new Point(x0, y), new Point(x1, y));
    }

    private static Brush Freeze(Brush b) { b.Freeze(); return b; }
}
