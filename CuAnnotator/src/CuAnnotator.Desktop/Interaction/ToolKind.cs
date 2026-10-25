namespace CuAnnotator.Desktop.Interaction;

/// <summary>Left toolbox tools (section 6.1 B).</summary>
public enum ToolKind
{
    Select,
    Rectangle,
    Polygon,
    Lasso,
    SmartContour,
    ForegroundBrush,
    BackgroundBrush,
    BoundaryBrush,
    Eraser,
    VertexEditor,
    Split,
    Merge,
    Fill,
    ColorSelect,
    MagicWand,
    Measure,
}

public enum OverlayMode
{
    /// <summary>Masks and contours drawn over the image.</summary>
    Normal,
    /// <summary>"Before": image only.</summary>
    Hidden,
    /// <summary>Overlays only on the right half of the viewport (split view).</summary>
    Split,
    /// <summary>Overlays in alternating checkerboard cells (boundary inspection).</summary>
    Checkerboard,
    /// <summary>High-contrast contours only, no fill.</summary>
    ContoursOnly,
}
