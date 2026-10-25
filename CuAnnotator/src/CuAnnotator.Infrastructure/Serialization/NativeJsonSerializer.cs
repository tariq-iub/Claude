using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using CuAnnotator.Application.Services;
using CuAnnotator.Contracts;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Geometry;

namespace CuAnnotator.Infrastructure.Serialization;

/// <summary>
/// Native, lossless JSON format (schema v1). Masks are stored as COCO compressed RLE relative to
/// their bounds; polygons as [[x,y],…] arrays; enums as strings; timestamps ISO 8601 with offset (UTC).
/// </summary>
public sealed class NativeJsonSerializer : IProjectSerializer
{
    public static JsonSerializerOptions Options { get; } = CreateOptions();

    private static JsonSerializerOptions CreateOptions()
    {
        var o = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        };
        o.Converters.Add(new JsonStringEnumConverter());
        o.Converters.Add(new RasterMaskConverter());
        o.Converters.Add(new PixelPointConverter());
        o.Converters.Add(new PixelRectConverter());
        return o;
    }

    public string Serialize(AnnotationProject project)
    {
        var envelope = new JsonObject
        {
            ["format"] = NativeFormat.FormatName,
            ["schemaVersion"] = AnnotationProject.CurrentSchemaVersion,
            ["appVersion"] = AppInfo.Version,
            ["savedAt"] = DateTimeOffset.UtcNow.ToString("O"),
            ["coordinateConvention"] = NativeFormat.CoordinateConvention,
            ["rasterizationRule"] = NativeFormat.RasterizationRule,
            ["project"] = JsonSerializer.SerializeToNode(project, Options),
        };
        return envelope.ToJsonString(Options);
    }

    public AnnotationProject Deserialize(string json)
    {
        var root = JsonNode.Parse(json) ?? throw new InvalidDataException("Empty project file.");
        CheckEnvelope(root, NativeFormat.FormatName);
        var project = root["project"].Deserialize<AnnotationProject>(Options) ?? throw new InvalidDataException("Project payload missing.");
        Migrate(project);
        return project;
    }

    public string SerializeImage(AnnotationProject project, ImageItem image)
    {
        var envelope = new JsonObject
        {
            ["format"] = NativeFormat.ImageAnnotationFormatName,
            ["schemaVersion"] = AnnotationProject.CurrentSchemaVersion,
            ["appVersion"] = AppInfo.Version,
            ["savedAt"] = DateTimeOffset.UtcNow.ToString("O"),
            ["project"] = project.Name,
            ["projectId"] = project.Id.ToString(),
            ["coordinateConvention"] = NativeFormat.CoordinateConvention,
            ["rasterizationRule"] = NativeFormat.RasterizationRule,
            ["taxonomy"] = JsonSerializer.SerializeToNode(project.Taxonomy, Options),
            ["image"] = JsonSerializer.SerializeToNode(image, Options),
        };
        return envelope.ToJsonString(Options);
    }

    public (ImageItem Image, Taxonomy Taxonomy) DeserializeImage(string json)
    {
        var root = JsonNode.Parse(json) ?? throw new InvalidDataException("Empty file.");
        CheckEnvelope(root, NativeFormat.ImageAnnotationFormatName);
        return (root["image"].Deserialize<ImageItem>(Options)!, root["taxonomy"].Deserialize<Taxonomy>(Options)!);
    }

    private static void CheckEnvelope(JsonNode root, string format)
    {
        if ((string?)root["format"] != format) throw new InvalidDataException($"Not a {format} file.");
        int v = (int?)root["schemaVersion"] ?? 0;
        if (v > AnnotationProject.CurrentSchemaVersion)
            throw new InvalidDataException($"File uses schema v{v}; this version of CuAnnotator supports up to v{AnnotationProject.CurrentSchemaVersion}. Upgrade the application.");
        if (v < 1) throw new InvalidDataException("Missing or invalid schemaVersion.");
    }

    /// <summary>Upgrades older schema versions in place (none exist yet; v1 is the first release).</summary>
    private static void Migrate(AnnotationProject p)
    {
        p.SchemaVersion = AnnotationProject.CurrentSchemaVersion;
        foreach (var img in p.Images)
            foreach (var a in img.Annotations)
                a.ImageId = img.Id;
    }

    private sealed class RasterMaskConverter : JsonConverter<RasterMask>
    {
        public override RasterMask Read(ref Utf8JsonReader reader, Type t, JsonSerializerOptions o)
        {
            using var doc = JsonDocument.ParseValue(ref reader);
            var b = doc.RootElement.GetProperty("bounds");
            var bounds = new PixelRect(b[0].GetInt32(), b[1].GetInt32(), b[2].GetInt32(), b[3].GetInt32());
            if (bounds.IsEmpty) return RasterMask.Empty;
            var counts = CocoRle.FromCompressedString(doc.RootElement.GetProperty("rle").GetString() ?? "");
            var mask = CocoRle.DecodeLocal(counts, bounds);
            if (doc.RootElement.TryGetProperty("pixels", out var px) && px.GetInt64() != mask.PixelCount)
                throw new InvalidDataException($"Mask pixel count mismatch ({mask.PixelCount} decoded vs {px.GetInt64()} recorded): file is corrupted.");
            return mask;
        }

        public override void Write(Utf8JsonWriter w, RasterMask value, JsonSerializerOptions o)
        {
            var m = value.Trim();
            w.WriteStartObject();
            w.WriteStartArray("bounds");
            w.WriteNumberValue(m.Bounds.X); w.WriteNumberValue(m.Bounds.Y); w.WriteNumberValue(m.Bounds.Width); w.WriteNumberValue(m.Bounds.Height);
            w.WriteEndArray();
            w.WriteNumber("pixels", m.PixelCount);
            w.WriteString("rle", m.Bounds.IsEmpty ? "" : CocoRle.ToCompressedString(CocoRle.EncodeLocal(m)));
            w.WriteEndObject();
        }
    }

    private sealed class PixelPointConverter : JsonConverter<PixelPoint>
    {
        public override PixelPoint Read(ref Utf8JsonReader r, Type t, JsonSerializerOptions o)
        {
            r.Read(); int x = r.GetInt32(); r.Read(); int y = r.GetInt32(); r.Read();
            return new PixelPoint(x, y);
        }

        public override void Write(Utf8JsonWriter w, PixelPoint v, JsonSerializerOptions o)
        {
            w.WriteStartArray(); w.WriteNumberValue(v.X); w.WriteNumberValue(v.Y); w.WriteEndArray();
        }
    }

    private sealed class PixelRectConverter : JsonConverter<PixelRect>
    {
        public override PixelRect Read(ref Utf8JsonReader r, Type t, JsonSerializerOptions o)
        {
            var v = new int[4];
            r.Read();
            for (int i = 0; i < 4; i++) { v[i] = r.GetInt32(); r.Read(); }
            return new PixelRect(v[0], v[1], v[2], v[3]);
        }

        public override void Write(Utf8JsonWriter w, PixelRect v, JsonSerializerOptions o)
        {
            w.WriteStartArray(); w.WriteNumberValue(v.X); w.WriteNumberValue(v.Y); w.WriteNumberValue(v.Width); w.WriteNumberValue(v.Height); w.WriteEndArray();
        }
    }
}
