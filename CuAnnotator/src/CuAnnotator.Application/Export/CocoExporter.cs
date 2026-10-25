using CuAnnotator.Contracts.Coco;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Application.Export;

public enum CocoSegmentationMode
{
    /// <summary>Polygons when exact (no holes), RLE otherwise.</summary>
    Auto,
    /// <summary>Always compressed RLE (always exact).</summary>
    AlwaysRle,
}

/// <summary>
/// COCO instance export. Area is the exact mask pixel count; bbox is [x, y, w, h] of the mask in
/// pixel-edge coordinates. Polygon coordinates are pixel centres (x + 0.5, y + 0.5) of boundary pixels,
/// which is the convention COCO tools expect when rasterising. Annotations with holes are written as
/// compressed RLE (pycocotools-compatible) because COCO polygons cannot express holes.
/// Note: pycocotools rasterises polygons with its own rule, so polygon mode can differ from the native
/// mask by boundary pixels; <see cref="CocoSegmentationMode.AlwaysRle"/> is pixel-exact.
/// </summary>
public static class CocoExporter
{
    public static CocoDataset Build(AnnotationProject project, IEnumerable<(ImageItem Image, string FileName)> images,
        CocoSegmentationMode mode, ExportReport report, string annotationSet = "default")
    {
        var ds = new CocoDataset
        {
            Info = new CocoInfo
            {
                Description = $"{project.Name} — exported by CuAnnotator",
                Year = DateTime.UtcNow.Year,
                Contributor = project.Researcher,
                DateCreated = DateTimeOffset.UtcNow.ToString("O"),
            },
            Licenses = { new CocoLicense { Id = 0, Name = "Unspecified" } },
        };
        foreach (var c in project.Taxonomy.Classes.Where(c => c.Id != Taxonomy.UnlabeledId && c.Id != Taxonomy.IgnoreId).OrderBy(c => c.Id))
            ds.Categories.Add(new CocoCategory { Id = c.Id, Name = c.Name, Supercategory = c.ParentCategory?.Split('/').Reverse().Skip(1).FirstOrDefault() ?? "copper" });

        long imageId = 0, annId = 0;
        foreach (var (img, fileName) in images)
        {
            imageId++;
            ds.Images.Add(new CocoImage { Id = imageId, FileName = fileName, Width = img.Width, Height = img.Height, DateCaptured = img.ImportedAt.ToString("O") });
            foreach (var a in img.Annotations.Where(a => a.AnnotationSet == annotationSet).OrderBy(a => a.ZOrder))
            {
                if (a.ClassId == Taxonomy.IgnoreId)
                {
                    report.Info($"{img.FileName}: annotation {a.Id} has class Ignore (255); excluded from COCO (present in semantic masks as 255).");
                    continue;
                }
                var mask = a.Mask.ClipToImage(img.Width, img.Height);
                if (mask.IsEmpty) { report.Warn($"{img.FileName}: annotation {a.Id} is empty and was skipped."); continue; }
                bool hasHoles = a.Polygons.Any(p => p.Holes.Count > 0);
                object seg;
                if (mode == CocoSegmentationMode.AlwaysRle || hasHoles || a.Polygons.Count == 0 || a.Polygons.Any(p => p.Outer.Count < 3))
                {
                    seg = new CocoRleSegmentation
                    {
                        Size = new[] { img.Height, img.Width },
                        Counts = CocoRle.ToCompressedString(CocoRle.Encode(mask, img.Width, img.Height)),
                    };
                    if (hasHoles && mode == CocoSegmentationMode.Auto)
                        report.Info($"{img.FileName}: annotation {a.Id} has holes → exported as RLE.");
                }
                else
                {
                    seg = a.Polygons.Select(p => p.Outer.SelectMany(v => new[] { v.X + 0.5, v.Y + 0.5 }).ToArray()).ToList();
                }
                var bb = mask.ContentBounds();
                ds.Annotations.Add(new CocoAnnotation
                {
                    Id = ++annId, ImageId = imageId, CategoryId = a.ClassId, Segmentation = seg,
                    Area = mask.PixelCount, BBox = new double[] { bb.X, bb.Y, bb.Width, bb.Height }, IsCrowd = 0,
                    Extension = new CocoExtension { AnnotationId = a.Id.ToString(), VerificationStatus = a.VerificationStatus.ToString(), Revision = a.Revision },
                });
            }
        }
        return ds;
    }
}
