using System.Text;
using System.Text.Json;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Infrastructure.Serialization;

namespace CuAnnotator.Infrastructure.Persistence;

/// <summary>
/// Append-only audit trail (one JSON object per line). Each entry is flushed immediately so the log
/// survives crashes independently of project saves. Actor prefixes: <c>user:</c>, <c>algorithm:</c>, <c>system</c>.
/// </summary>
public sealed class JsonlAuditLog : IAuditLog
{
    private readonly string _path;
    private readonly object _gate = new();

    public JsonlAuditLog(string path) { _path = path; Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!); }

    public void Append(AuditEntry entry)
    {
        var line = JsonSerializer.Serialize(entry, NativeJsonSerializer.Options).ReplaceLineEndings(" ");
        lock (_gate)
        {
            using var fs = new FileStream(_path, FileMode.Append, FileAccess.Write, FileShare.Read);
            var bytes = Encoding.UTF8.GetBytes(Compact(line) + "\n");
            fs.Write(bytes);
            fs.Flush(true);
        }
    }

    public IReadOnlyList<AuditEntry> ReadAll()
    {
        if (!File.Exists(_path)) return Array.Empty<AuditEntry>();
        var list = new List<AuditEntry>();
        lock (_gate)
            foreach (var line in File.ReadLines(_path))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try { if (JsonSerializer.Deserialize<AuditEntry>(line, NativeJsonSerializer.Options) is { } e) list.Add(e); }
                catch (JsonException) { /* a torn final line after a crash is skipped */ }
            }
        return list;
    }

    private static string Compact(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return JsonSerializer.Serialize(doc.RootElement);
    }

    /// <summary>Human-readable rendering in the format of the requirements (timestamp, image, action, details).</summary>
    public static string Format(IEnumerable<AuditEntry> entries, Func<Guid, string>? imageName = null)
    {
        var sb = new StringBuilder();
        foreach (var e in entries)
        {
            sb.AppendLine(e.Timestamp.UtcDateTime.ToString("yyyy-MM-dd HH:mm:ss") + " UTC");
            if (e.ImageId is { } id) sb.AppendLine($"Image: {imageName?.Invoke(id) ?? id.ToString()}");
            sb.AppendLine($"Action: {e.Action}  [{e.Actor}]");
            if (!string.IsNullOrEmpty(e.Details)) sb.AppendLine(e.Details);
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

/// <summary>In-memory audit log for tests and transient sessions.</summary>
public sealed class MemoryAuditLog : IAuditLog
{
    private readonly List<AuditEntry> _entries = new();
    public void Append(AuditEntry entry) { lock (_entries) _entries.Add(entry); }
    public IReadOnlyList<AuditEntry> ReadAll() { lock (_entries) return _entries.ToList(); }
}
