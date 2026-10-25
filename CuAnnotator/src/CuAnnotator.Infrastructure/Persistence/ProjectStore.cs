using CuAnnotator.Application.Services;
using CuAnnotator.Contracts;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Infrastructure.FileStorage;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Infrastructure.SQLite;
using Serilog;

namespace CuAnnotator.Infrastructure.Persistence;

/// <summary>
/// Project folder layout:
/// <code>
/// MyProject/
///   project.cuann.json        authoritative native project (atomic replace)
///   project.cuann.json.bak    previous valid version
///   project.autosave.json     recovery data written by autosave (deleted after a manual save)
///   index.db                  SQLite index (rebuilt on save)
///   audit.jsonl               append-only audit trail
///   images/                   imported originals (byte-identical copies)
///   exports/
/// </code>
/// </summary>
public sealed class ProjectStore(IProjectSerializer serializer, ILogger? logger = null) : IProjectStore
{
    public const string AutosaveFileName = "project.autosave.json";
    private readonly ILogger _log = logger ?? Serilog.Core.Logger.None;

    /// <summary>Set when the last load had to fall back to the backup file.</summary>
    public string? LastLoadWarning { get; private set; }

    public async Task SaveAsync(AnnotationProject project, string folder, CancellationToken ct = default)
    {
        Directory.CreateDirectory(folder);
        var json = serializer.Serialize(project);
        serializer.Deserialize(json); // self-check: never replace a valid file with something we cannot read back
        await AtomicFile.WriteAllTextAsync(Path.Combine(folder, NativeFormat.ProjectFileName), json, Path.Combine(folder, NativeFormat.BackupFileName), ct);
        try
        {
            new SqliteProjectIndex(Path.Combine(folder, NativeFormat.IndexDatabaseName)).Rebuild(project, OpenAuditLog(folder).ReadAll());
        }
        catch (Exception ex) when (ex is Microsoft.Data.Sqlite.SqliteException or IOException)
        {
            _log.Warning(ex, "SQLite index rebuild failed; the JSON project was saved successfully");
        }
        DiscardRecovery(folder);
        _log.Information("Saved project {Name} to {Folder} ({Images} images, {Annotations} annotations)",
            project.Name, folder, project.Images.Count, project.Images.Sum(i => i.Annotations.Count));
    }

    public async Task<AnnotationProject> LoadAsync(string folder, CancellationToken ct = default)
    {
        LastLoadWarning = null;
        AtomicFile.CleanupTemporaryFiles(folder);
        var main = Path.Combine(folder, NativeFormat.ProjectFileName);
        var bak = Path.Combine(folder, NativeFormat.BackupFileName);
        if (!File.Exists(main) && !File.Exists(bak)) throw new FileNotFoundException("No CuAnnotator project in this folder.", main);
        try
        {
            return serializer.Deserialize(await File.ReadAllTextAsync(main, ct));
        }
        catch (Exception ex) when ((ex is System.Text.Json.JsonException or InvalidDataException or FileNotFoundException) && File.Exists(bak))
        {
            _log.Error(ex, "Project file unreadable; loading backup {Backup}", bak);
            LastLoadWarning = $"The project file could not be read ({ex.Message}). The previous version from the backup was loaded instead.";
            return serializer.Deserialize(await File.ReadAllTextAsync(bak, ct));
        }
    }

    public bool HasRecoveryData(string folder)
    {
        var a = Path.Combine(folder, AutosaveFileName);
        var m = Path.Combine(folder, NativeFormat.ProjectFileName);
        return File.Exists(a) && (!File.Exists(m) || File.GetLastWriteTimeUtc(a) > File.GetLastWriteTimeUtc(m));
    }

    public Task SaveRecoveryAsync(AnnotationProject project, string folder, CancellationToken ct = default) =>
        AtomicFile.WriteAllTextAsync(Path.Combine(folder, AutosaveFileName), serializer.Serialize(project), null, ct);

    public async Task<AnnotationProject?> LoadRecoveryAsync(string folder, CancellationToken ct = default)
    {
        var a = Path.Combine(folder, AutosaveFileName);
        if (!File.Exists(a)) return null;
        try { return serializer.Deserialize(await File.ReadAllTextAsync(a, ct)); }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidDataException)
        {
            _log.Warning(ex, "Recovery file unreadable");
            return null;
        }
    }

    public void DiscardRecovery(string folder)
    {
        var a = Path.Combine(folder, AutosaveFileName);
        if (File.Exists(a)) File.Delete(a);
    }

    public IAuditLog OpenAuditLog(string folder) => new JsonlAuditLog(Path.Combine(folder, NativeFormat.AuditLogFileName));

    /// <summary>Save As: copies images and the audit trail, then saves the project in the new folder.</summary>
    public async Task SaveAsAsync(AnnotationProject project, string oldFolder, string newFolder, CancellationToken ct = default)
    {
        Directory.CreateDirectory(newFolder);
        foreach (var img in project.Images)
        {
            var src = ProjectPaths.Resolve(oldFolder, img.RelativePath);
            if (Path.IsPathRooted(img.RelativePath)) continue; // external reference stays absolute
            var dst = ProjectPaths.Resolve(newFolder, img.RelativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(dst)!);
            if (File.Exists(src) && !File.Exists(dst)) File.Copy(src, dst);
        }
        var audit = Path.Combine(oldFolder, NativeFormat.AuditLogFileName);
        if (File.Exists(audit)) File.Copy(audit, Path.Combine(newFolder, NativeFormat.AuditLogFileName), true);
        await SaveAsync(project, newFolder, ct);
    }
}
