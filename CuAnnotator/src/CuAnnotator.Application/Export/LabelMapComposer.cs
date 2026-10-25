using CuAnnotator.Domain.Entities;

namespace CuAnnotator.Application.Export;

public sealed class LabelMap
{
    public required int Width { get; init; }
    public required int Height { get; init; }
    /// <summary>Class ID per pixel (row-major, single channel).</summary>
    public required byte[] Labels { get; init; }
    /// <summary>Pixels claimed by more than one annotation (resolved by Z order).</summary>
    public long OverlapPixels { get; init; }
}

/// <summary>
/// Builds the semantic label map of an image from its annotations: pixels default to 0 (Unlabeled);
/// annotations are painted in ascending <see cref="Annotation.ZOrder"/> (ties by creation time) so
/// the result is deterministic. Hidden annotations are still exported (visibility is a display setting).
/// </summary>
public static class LabelMapComposer
{
    public static LabelMap Compose(ImageItem image, string annotationSet = "default", Func<Annotation, bool>? filter = null)
    {
        int w = image.Width, h = image.Height;
        var labels = new byte[w * h];
        var claimed = new bool[w * h];
        long overlap = 0;
        foreach (var a in image.Annotations.Where(a => a.AnnotationSet == annotationSet && (filter?.Invoke(a) ?? true))
                     .OrderBy(a => a.ZOrder).ThenBy(a => a.CreatedAt))
        {
            if (a.ClassId is < 0 or > 255) throw new InvalidOperationException($"Annotation {a.Id} has class {a.ClassId}, not representable in 8 bits.");
            var m = a.Mask;
            var b = m.Bounds.ClampTo(w, h);
            for (int y = b.Y; y < b.Bottom; y++)
            {
                int mrow = (y - m.Bounds.Y) * m.Width;
                for (int x = b.X; x < b.Right; x++)
                {
                    if (m.Data[mrow + x - m.Bounds.X] == 0) continue;
                    int i = y * w + x;
                    if (claimed[i]) overlap++;
                    claimed[i] = true;
                    labels[i] = (byte)a.ClassId;
                }
            }
        }
        return new LabelMap { Width = w, Height = h, Labels = labels, OverlapPixels = overlap };
    }

    /// <summary>Binary 0/255 mask of the union of annotations matching <paramref name="filter"/>.</summary>
    public static byte[] Binary(ImageItem image, Func<Annotation, bool> filter, string annotationSet = "default")
    {
        var result = new byte[image.Width * image.Height];
        foreach (var a in image.Annotations.Where(a => a.AnnotationSet == annotationSet && filter(a)))
        {
            var m = a.Mask; var b = m.Bounds.ClampTo(image.Width, image.Height);
            for (int y = b.Y; y < b.Bottom; y++)
                for (int x = b.X; x < b.Right; x++)
                    if (m.Data[(y - m.Bounds.Y) * m.Width + x - m.Bounds.X] != 0) result[y * image.Width + x] = 255;
        }
        return result;
    }
}
