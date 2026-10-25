using System.Text.Json.Serialization;

namespace CuAnnotator.Contracts.Coco;

/// <summary>COCO "instances" file (https://cocodataset.org/#format-data).</summary>
public sealed class CocoDataset
{
    [JsonPropertyName("info")] public CocoInfo Info { get; set; } = new();
    [JsonPropertyName("licenses")] public List<CocoLicense> Licenses { get; set; } = new();
    [JsonPropertyName("images")] public List<CocoImage> Images { get; set; } = new();
    [JsonPropertyName("annotations")] public List<CocoAnnotation> Annotations { get; set; } = new();
    [JsonPropertyName("categories")] public List<CocoCategory> Categories { get; set; } = new();
}

public sealed class CocoInfo
{
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("version")] public string Version { get; set; } = "1.0";
    [JsonPropertyName("year")] public int Year { get; set; }
    [JsonPropertyName("contributor")] public string Contributor { get; set; } = "";
    [JsonPropertyName("date_created")] public string DateCreated { get; set; } = "";
}

public sealed class CocoLicense
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("url")] public string Url { get; set; } = "";
}

public sealed class CocoImage
{
    [JsonPropertyName("id")] public long Id { get; set; }
    [JsonPropertyName("file_name")] public string FileName { get; set; } = "";
    [JsonPropertyName("width")] public int Width { get; set; }
    [JsonPropertyName("height")] public int Height { get; set; }
    [JsonPropertyName("license")] public int License { get; set; }
    [JsonPropertyName("date_captured")] public string DateCaptured { get; set; } = "";
}

public sealed class CocoCategory
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("supercategory")] public string Supercategory { get; set; } = "";
}

/// <summary>
/// One instance. <see cref="Segmentation"/> is either a list of polygons (<c>[[x1,y1,x2,y2,…],…]</c>)
/// or a compressed RLE object <c>{"size":[h,w],"counts":"…"}</c> (iscrowd = 0 in both cases here).
/// </summary>
public sealed class CocoAnnotation
{
    [JsonPropertyName("id")] public long Id { get; set; }
    [JsonPropertyName("image_id")] public long ImageId { get; set; }
    [JsonPropertyName("category_id")] public int CategoryId { get; set; }
    [JsonPropertyName("segmentation")] public object Segmentation { get; set; } = new();
    [JsonPropertyName("area")] public double Area { get; set; }
    [JsonPropertyName("bbox")] public double[] BBox { get; set; } = new double[4];
    [JsonPropertyName("iscrowd")] public int IsCrowd { get; set; }
    /// <summary>Non-standard extension (ignored by COCO tools): links back to the native annotation.</summary>
    [JsonPropertyName("cuannotator")] public CocoExtension? Extension { get; set; }
}

public sealed class CocoRleSegmentation
{
    [JsonPropertyName("size")] public int[] Size { get; set; } = new int[2];
    [JsonPropertyName("counts")] public string Counts { get; set; } = "";
}

public sealed class CocoExtension
{
    [JsonPropertyName("annotation_id")] public string AnnotationId { get; set; } = "";
    [JsonPropertyName("verification_status")] public string VerificationStatus { get; set; } = "";
    [JsonPropertyName("revision")] public int Revision { get; set; }
}
