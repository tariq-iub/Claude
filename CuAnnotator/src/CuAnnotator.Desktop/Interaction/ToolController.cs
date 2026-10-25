using System.Windows.Input;
using System.Windows.Media;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Desktop.Controls;
using CuAnnotator.Desktop.Rendering;
using CuAnnotator.Desktop.ViewModels;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Vision.Segmentation;

namespace CuAnnotator.Desktop.Interaction;

/// <summary>
/// Translates pointer input (already converted to image coordinates by <see cref="ImageViewport"/>) into
/// editor operations for the active tool. Holds only transient gesture state; all model changes go through
/// <see cref="EditorViewModel"/> / <see cref="AnnotationSession"/>.
/// </summary>
public sealed class ToolController
{
    private readonly ImageViewport _vp;
    private readonly MainViewModel _main;
    private readonly List<ImagePoint> _points = new();
    private ImagePoint? _start;
    private System.Windows.Point _startView;
    private bool _active;
    private (int C, int R, int I)? _dragVertex;
    private Guid? _editingId;
    private ImagePoint? _moveStart;

    public ToolController(ImageViewport viewport, MainViewModel main)
    {
        _vp = viewport; _main = main;
        _vp.ImagePointerDown += OnDown;
        _vp.ImagePointerMove += OnMove;
        _vp.ImagePointerUp += OnUp;
        main.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(MainViewModel.ActiveTool) or nameof(MainViewModel.Editor)) { Reset(); ConfigureForTool(); }
        };
        ConfigureForTool();
    }

    private EditorViewModel? Editor => _main.Editor;
    private ToolKind Tool => _main.ActiveTool;
    private int Radius => _main.BrushRadius;

    private void ConfigureForTool()
    {
        _vp.PanWithLeftButton = Tool == ToolKind.Select;
        _vp.Cursor = Tool switch
        {
            ToolKind.Select => Cursors.Arrow,
            ToolKind.VertexEditor => Cursors.Hand,
            ToolKind.Measure => Cursors.Cross,
            _ => Cursors.Cross,
        };
        if (Tool == ToolKind.VertexEditor) BeginVertexEditing();
        else if (Editor is not null) { Editor.Scene.EditPolygons = null; Editor.Scene.Invalidate(); }
    }

    public void Reset()
    {
        _points.Clear(); _start = null; _active = false; _dragVertex = null; _moveStart = null;
        if (Editor is { } e) { e.Scene.Preview.Clear(); e.Scene.Invalidate(); }
    }

    private void SetPreview(params PreviewShape[] shapes)
    {
        if (Editor is not { } e) return;
        e.Scene.Preview.Clear();
        e.Scene.Preview.AddRange(shapes);
        e.Scene.Invalidate();
    }

    private static List<PixelPoint> ToPixels(IEnumerable<ImagePoint> pts)
    {
        var list = new List<PixelPoint>();
        foreach (var p in pts.Select(p => p.ToPixel())) if (list.Count == 0 || list[^1] != p) list.Add(p);
        return list;
    }

    private static PreviewShape RectShape(ImagePoint a, ImagePoint b, Color c)
    {
        var pa = a.ToPixel(); var pb = b.ToPixel();
        var r = PixelRect.FromInclusivePixels(pa, pb);
        return new PreviewShape
        {
            Points = new[] { new ImagePoint(r.X, r.Y), new ImagePoint(r.Right, r.Y), new ImagePoint(r.Right, r.Bottom), new ImagePoint(r.X, r.Bottom) },
            Closed = true, Color = c, ScreenThickness = 1.5, Dashed = true,
        };
    }

    // ───────────── Pointer handlers ─────────────

    private void OnDown(object? sender, ImagePointerEventArgs e)
    {
        if (Editor is not { } ed) return;
        if (e.Button == MouseButton.Right && Tool != ToolKind.VertexEditor) return;
        _start = e.Image; _startView = e.View; _active = true;
        switch (Tool)
        {
            case ToolKind.Polygon:
                if (e.ClickCount >= 2) { ClosePolygon(); return; }
                _points.Add(e.Image);
                UpdatePolygonPreview(e.Image);
                break;
            case ToolKind.Lasso:
            case ToolKind.ForegroundBrush:
            case ToolKind.BackgroundBrush:
            case ToolKind.BoundaryBrush:
            case ToolKind.Eraser:
            case ToolKind.Split:
            case ToolKind.Measure:
                _points.Clear();
                _points.Add(e.Image);
                break;
            case ToolKind.VertexEditor:
                VertexDown(ed, e);
                break;
        }
    }

    private void OnMove(object? sender, ImagePointerEventArgs e)
    {
        if (Editor is not { } ed) return;
        ed.UpdateCursor(e.Image);
        if (Tool == ToolKind.Polygon && _points.Count > 0) { UpdatePolygonPreview(e.Image); return; }
        if (!_active || _start is not { } s) { HoverPreview(e); return; }
        switch (Tool)
        {
            case ToolKind.Rectangle:
            case ToolKind.SmartContour:
                SetPreview(RectShape(s, e.Image, Tool == ToolKind.SmartContour ? Colors.Orange : Colors.DeepSkyBlue));
                break;
            case ToolKind.Lasso:
                AddPoint(e.Image);
                SetPreview(new PreviewShape { Points = _points.ToList(), Closed = true, Color = Colors.DeepSkyBlue, Dashed = true });
                break;
            case ToolKind.ForegroundBrush:
            case ToolKind.BackgroundBrush:
            case ToolKind.Eraser:
                AddPoint(e.Image);
                var color = Tool == ToolKind.ForegroundBrush ? Color.FromArgb(150, 40, 200, 80) : Tool == ToolKind.Eraser ? Color.FromArgb(150, 255, 255, 255) : Color.FromArgb(150, 230, 50, 50);
                SetPreview(new PreviewShape { Points = _points.Select(p => ImagePoint.CenterOf(p.ToPixel())).ToList(), Color = color, ImageThickness = 2 * Radius + 1 });
                break;
            case ToolKind.BoundaryBrush:
            case ToolKind.Split:
                AddPoint(e.Image);
                SetPreview(new PreviewShape { Points = _points.ToList(), Color = Tool == ToolKind.Split ? Colors.OrangeRed : Colors.Gold, ScreenThickness = 2 });
                break;
            case ToolKind.Measure:
                _points.Clear(); _points.Add(s); _points.Add(e.Image);
                SetPreview(new PreviewShape { Points = _points.ToList(), Color = Colors.Yellow, ScreenThickness = 1.5 });
                ed.SegmentationStatus = MeasureText(ed, s, e.Image);
                break;
            case ToolKind.VertexEditor:
                VertexMove(ed, e);
                break;
        }
    }

    private void OnUp(object? sender, ImagePointerEventArgs e)
    {
        if (Editor is not { } ed || !_active || _start is not { } s) return;
        _active = false;
        bool click = (e.View - _startView).Length < 4;
        switch (Tool)
        {
            case ToolKind.Select:
                if (click) ed.Select(ed.HitTest(e.Pixel)?.Id, (e.Modifiers & ModifierKeys.Control) != 0);
                break;
            case ToolKind.Merge:
                ed.Select(ed.HitTest(e.Pixel)?.Id, additive: true);
                ed.SegmentationStatus = $"{ed.SelectedIds.Count} selected for merging. Press Enter (or Merge) to merge.";
                break;
            case ToolKind.Rectangle:
            case ToolKind.SmartContour:
                SetPreview();
                if (click) { ed.Select(ed.HitTest(e.Pixel)?.Id, (e.Modifiers & ModifierKeys.Control) != 0); break; }
                ed.BeginRoi(PixelRect.FromInclusivePixels(s.ToPixel(), e.Pixel), Tool == ToolKind.SmartContour ? SegmentationAlgorithmKind.Hybrid : null);
                break;
            case ToolKind.Lasso:
                SetPreview();
                if (_points.Count >= 3) ed.BeginRegion(ToPixels(_points), "Lasso");
                break;
            case ToolKind.ForegroundBrush:
            case ToolKind.BackgroundBrush:
            case ToolKind.Eraser:
            {
                SetPreview();
                var stroke = new Stroke(ToPixels(_points), Radius);
                if (Tool == ToolKind.Eraser) ed.BrushOnSelection(stroke, add: false);
                else if (ed.Session.Candidate is not null)
                    ed.AddGuidance(Tool == ToolKind.ForegroundBrush ? GuidanceKind.Foreground : GuidanceKind.Background, stroke);
                else ed.BrushOnSelection(stroke, add: Tool == ToolKind.ForegroundBrush);
                break;
            }
            case ToolKind.BoundaryBrush:
            {
                var path = ToPixels(_points);
                if (path.Count < 2) { SetPreview(); break; }
                var snapped = ed.SnapBoundary(path);
                SetPreview();
                ed.AddGuidance(GuidanceKind.Barrier, new Stroke(snapped, 1));
                break;
            }
            case ToolKind.Split:
                SetPreview();
                if (_points.Count >= 2) ed.Split(ToPixels(_points));
                break;
            case ToolKind.Fill:
                ed.FillAt(e.Pixel);
                break;
            case ToolKind.MagicWand:
                _ = ed.MagicWandAsync(e.Pixel);
                break;
            case ToolKind.ColorSelect:
            {
                var v = _vp.VisibleImageRect();
                var within = new PixelRect((int)v.X, (int)v.Y, (int)Math.Ceiling(v.Width), (int)Math.Ceiling(v.Height)).ClampTo(ed.Image.Width, ed.Image.Height);
                _ = ed.ColorSelectAsync(e.Pixel, ed.Session.Candidate?.Roi ?? within);
                break;
            }
            case ToolKind.Measure:
                ed.SegmentationStatus = MeasureText(ed, s, e.Image);
                break;
            case ToolKind.VertexEditor:
                VertexUp(ed, e);
                break;
        }
        if (Tool != ToolKind.Polygon) _points.Clear();
    }

    private void AddPoint(ImagePoint p)
    {
        if (_points.Count == 0 || Math.Abs(_points[^1].X - p.X) + Math.Abs(_points[^1].Y - p.Y) >= 0.75 / Math.Max(0.05, _vp.Transform.Scale))
            _points.Add(p);
    }

    private void HoverPreview(ImagePointerEventArgs e)
    {
        if (Tool is ToolKind.ForegroundBrush or ToolKind.BackgroundBrush or ToolKind.Eraser)
            SetPreview(new PreviewShape { Points = new[] { ImagePoint.CenterOf(e.Pixel) }, Color = Color.FromArgb(90, 255, 255, 255), ImageThickness = 2 * Radius + 1 });
        else if (Tool == ToolKind.VertexEditor && Editor?.Scene.EditPolygons is { } polys)
        {
            var hot = NearestVertex(polys, e.Image);
            if (hot != Editor.Scene.HotVertex) { Editor.Scene.HotVertex = hot; Editor.Scene.Invalidate(); }
        }
    }

    private static string MeasureText(EditorViewModel ed, ImagePoint a, ImagePoint b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y, d = Math.Sqrt(dx * dx + dy * dy);
        var um = ed.Image.MicrometersPerPixel is { } m ? $" = {d * m:0.##} µm" : " (no µm calibration for this image)";
        return $"Distance {d:0.##} px{um}; Δx {dx:0.#}, Δy {dy:0.#}, angle {Math.Atan2(-dy, dx) * 180 / Math.PI:0.#}°";
    }

    // ───────────── Polygon ROI ─────────────

    private void UpdatePolygonPreview(ImagePoint cursor) =>
        SetPreview(new PreviewShape { Points = _points.Append(cursor).ToList(), Closed = false, Color = Colors.DeepSkyBlue, ScreenThickness = 1.5, Dashed = true });

    public bool ClosePolygon()
    {
        if (Tool != ToolKind.Polygon || _points.Count < 3 || Editor is not { } ed) return false;
        var pts = ToPixels(_points);
        Reset();
        ed.BeginRegion(pts, "Polygon");
        return true;
    }

    // ───────────── Vertex editing ─────────────

    public void BeginVertexEditing()
    {
        if (Editor is not { } ed) return;
        if (ed.SelectedIds.Count != 1) { ed.Scene.EditPolygons = null; ed.SegmentationStatus = "Select one annotation to edit its vertices."; ed.Scene.Invalidate(); return; }
        var a = ed.Session.Find(ed.SelectedIds.Single());
        _editingId = a.Id;
        int vertices = a.Polygons.Sum(p => p.VertexCount);
        double eps = _main.Settings.EditSimplification;
        if (vertices > 400 && eps > 0)
        {
            ed.Scene.EditPolygons = a.Polygons.Select(p => new PolygonComponent(PolygonMath.SimplifyClosed(p.Outer, eps), p.Holes.Select(h => PolygonMath.SimplifyClosed(h, eps)))).ToList();
            ed.SegmentationStatus = $"Editing a simplified outline (ε = {eps} px, {ed.Scene.EditPolygons.Sum(p => p.VertexCount)} of {vertices} vertices). " +
                                    "The first edit re-rasterises the mask from this outline; the exact previous mask stays in the revision history.";
        }
        else
        {
            ed.Scene.EditPolygons = a.Polygons.Select(p => p.Clone()).ToList();
            ed.SegmentationStatus = $"Editing {vertices} exact boundary vertices.";
        }
        ed.Scene.Invalidate();
    }

    private (int, int, int)? NearestVertex(List<PolygonComponent> polys, ImagePoint p)
    {
        double tol = 8 / _vp.Transform.Scale, best = tol * tol;
        (int, int, int)? hit = null;
        for (int c = 0; c < polys.Count; c++)
        {
            var rings = polys[c].Holes.Prepend(polys[c].Outer).ToList();
            for (int r = 0; r < rings.Count; r++)
                for (int i = 0; i < rings[r].Count; i++)
                {
                    double dx = rings[r][i].X + 0.5 - p.X, dy = rings[r][i].Y + 0.5 - p.Y, d = dx * dx + dy * dy;
                    if (d < best) { best = d; hit = (c, r, i); }
                }
        }
        return hit;
    }

    private static List<PixelPoint> Ring(PolygonComponent c, int r) => r == 0 ? c.Outer : c.Holes[r - 1];

    private void VertexDown(EditorViewModel ed, ImagePointerEventArgs e)
    {
        if (ed.Scene.EditPolygons is null || _editingId is null || !ed.SelectedIds.Contains(_editingId.Value))
        {
            ed.Select(ed.HitTest(e.Pixel)?.Id, false);
            BeginVertexEditing();
            _active = false;
            return;
        }
        var polys = ed.Scene.EditPolygons;
        var hit = NearestVertex(polys, e.Image);
        if (e.Button == MouseButton.Right)
        {
            _active = false;
            if (hit is { } h && Ring(polys[h.Item1], h.Item2).Count > 3)
            {
                Ring(polys[h.Item1], h.Item2).RemoveAt(h.Item3);
                Commit(ed);
            }
            return;
        }
        if ((e.Modifiers & ModifierKeys.Control) != 0) { _moveStart = e.Image; return; }
        if (hit is not null) { _dragVertex = hit; return; }
        if (e.ClickCount >= 2) InsertVertex(ed, polys, e.Image);
        _active = false;
    }

    private void InsertVertex(EditorViewModel ed, List<PolygonComponent> polys, ImagePoint p)
    {
        double best = double.MaxValue; (int C, int R, int I)? at = null;
        var q = p.ToPixel();
        for (int c = 0; c < polys.Count; c++)
            for (int r = 0; r <= polys[c].Holes.Count; r++)
            {
                var ring = Ring(polys[c], r);
                for (int i = 0; i < ring.Count; i++)
                {
                    double d = PolygonMath.DistanceToSegment(q, ring[i], ring[(i + 1) % ring.Count]);
                    if (d < best) { best = d; at = (c, r, i + 1); }
                }
            }
        if (at is { } a && best <= 10 / _vp.Transform.Scale)
        {
            Ring(polys[a.C], a.R).Insert(a.I, q);
            Commit(ed);
        }
    }

    private void VertexMove(EditorViewModel ed, ImagePointerEventArgs e)
    {
        if (ed.Scene.EditPolygons is not { } polys) return;
        if (_dragVertex is { } v)
        {
            var px = new PixelPoint(Math.Clamp(e.Pixel.X, 0, ed.Image.Width - 1), Math.Clamp(e.Pixel.Y, 0, ed.Image.Height - 1));
            Ring(polys[v.C], v.R)[v.I] = px;
            ed.Scene.HotVertex = v;
            ed.Scene.Invalidate();
        }
        else if (_moveStart is { } m)
        {
            var d = (e.Pixel.X - m.ToPixel().X, e.Pixel.Y - m.ToPixel().Y);
            SetPreview(new PreviewShape { Points = new[] { m, e.Image }, Color = Colors.Yellow, Dashed = true });
            ed.SegmentationStatus = $"Move by ({d.Item1}, {d.Item2}) px";
        }
    }

    private void VertexUp(EditorViewModel ed, ImagePointerEventArgs e)
    {
        if (_dragVertex is not null) { _dragVertex = null; Commit(ed); }
        else if (_moveStart is { } m && _editingId is { } id)
        {
            SetPreview();
            int dx = e.Pixel.X - m.ToPixel().X, dy = e.Pixel.Y - m.ToPixel().Y;
            _moveStart = null;
            if (dx != 0 || dy != 0) { ed.Move(id, dx, dy); BeginVertexEditing(); }
        }
    }

    private void Commit(EditorViewModel ed)
    {
        if (_editingId is not { } id || ed.Scene.EditPolygons is not { } polys) return;
        ed.ApplyPolygonEdit(id, polys.Select(p => p.Clone()).ToList());
        BeginVertexEditing();
    }

    // ───────────── Keyboard ─────────────

    /// <summary>Enter: close polygon / merge / accept. Returns true when handled.</summary>
    public bool OnEnter()
    {
        if (ClosePolygon()) return true;
        if (Editor is not { } ed) return false;
        if (Tool == ToolKind.Merge && ed.MergeSelectedCommand.CanExecute(null)) { ed.MergeSelectedCommand.Execute(null); return true; }
        if (ed.AcceptCandidateCommand.CanExecute(null)) { ed.AcceptCandidateCommand.Execute(null); return true; }
        return false;
    }

    /// <summary>Escape: abort the current gesture, else discard the candidate.</summary>
    public bool OnEscape()
    {
        if (_points.Count > 0 || _active) { Reset(); return true; }
        if (Editor?.Session.Candidate is not null) { Editor.CancelSegmentationCommand.Execute(null); return true; }
        if (Editor?.SelectedIds.Count > 0) { Editor.Select(null, false); return true; }
        return false;
    }
}
