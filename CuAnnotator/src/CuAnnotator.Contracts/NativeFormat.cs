namespace CuAnnotator.Contracts;

/// <summary>Constants describing the native file format.</summary>
public static class NativeFormat
{
    public const string FormatName = "CuAnnotator.Project";
    public const string ImageAnnotationFormatName = "CuAnnotator.ImageAnnotations";
    public const string ProjectFileName = "project.cuann.json";
    public const string BackupFileName = "project.cuann.json.bak";
    public const string IndexDatabaseName = "index.db";
    public const string AuditLogFileName = "audit.jsonl";
    public const string ImagesFolder = "images";
    public const string ExportsFolder = "exports";

    public const string CoordinateConvention =
        "Origin top-left; X right; Y down; zero-based integer pixel indices. Pixel (x,y) covers [x,x+1)×[y,y+1). " +
        "Polygon vertices are boundary-pixel indices (pixel centres). Masks are stored as COCO compressed RLE " +
        "(column-major) relative to 'bounds' [x, y, width, height].";

    public const string RasterizationRule =
        "Components largest-first; outer ring scan-filled and boundary set (8-connected); each hole filled with 0 then " +
        "its boundary pixels set to 1. The mask, not the polygon, is authoritative.";
}
