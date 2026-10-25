using System.Text;

namespace CuAnnotator.Infrastructure.FileStorage;

/// <summary>
/// Crash-safe file replacement: content is written to a temporary file in the same directory and flushed
/// to disk, then swapped in with <see cref="File.Replace(string,string,string?)"/> (which keeps the previous
/// version as a backup) or moved into place. The previous valid file is never truncated before the new one
/// is completely on disk.
/// </summary>
public static class AtomicFile
{
    public static async Task WriteAllTextAsync(string path, string content, string? backupPath = null, CancellationToken ct = default)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path))!;
        Directory.CreateDirectory(dir);
        var tmp = Path.Combine(dir, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 16, FileOptions.WriteThrough | FileOptions.Asynchronous))
            {
                var bytes = new UTF8Encoding(false).GetBytes(content);
                await fs.WriteAsync(bytes, ct);
                await fs.FlushAsync(ct);
                fs.Flush(flushToDisk: true);
            }
            if (File.Exists(path))
            {
                if (backupPath is not null && File.Exists(backupPath)) File.Delete(backupPath);
                File.Replace(tmp, path, backupPath, ignoreMetadataErrors: true);
            }
            else File.Move(tmp, path);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    /// <summary>Deletes stale temporary files left by an interrupted save.</summary>
    public static int CleanupTemporaryFiles(string directory)
    {
        if (!Directory.Exists(directory)) return 0;
        int n = 0;
        foreach (var f in Directory.EnumerateFiles(directory, ".*.tmp"))
        {
            try { File.Delete(f); n++; } catch (IOException) { }
        }
        return n;
    }
}
