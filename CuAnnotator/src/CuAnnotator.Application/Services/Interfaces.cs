using CuAnnotator.Domain.Entities;

namespace CuAnnotator.Application.Services;

public static class AppInfo
{
    public static string Version { get; } =
        typeof(AppInfo).Assembly.GetName().Version?.ToString(3) ?? "1.0.0";
}

/// <summary>Native JSON (de)serialisation of projects and single-image annotation documents.</summary>
public interface IProjectSerializer
{
    string Serialize(AnnotationProject project);
    AnnotationProject Deserialize(string json);
    string SerializeImage(AnnotationProject project, ImageItem image);
    (ImageItem Image, Taxonomy Taxonomy) DeserializeImage(string json);
}

/// <summary>Project folder persistence (implemented in Infrastructure).</summary>
public interface IProjectStore
{
    /// <summary>Atomically saves the project; keeps the previous valid file as a backup.</summary>
    Task SaveAsync(AnnotationProject project, string projectFolder, CancellationToken ct = default);
    Task<AnnotationProject> LoadAsync(string projectFolder, CancellationToken ct = default);
    /// <summary>True if a recovery/autosave file newer than the project file exists.</summary>
    bool HasRecoveryData(string projectFolder);
    Task SaveRecoveryAsync(AnnotationProject project, string projectFolder, CancellationToken ct = default);
    Task<AnnotationProject?> LoadRecoveryAsync(string projectFolder, CancellationToken ct = default);
    void DiscardRecovery(string projectFolder);
    IAuditLog OpenAuditLog(string projectFolder);
}

public static class ProjectPaths
{
    public static string Resolve(string projectFolder, string relativeOrAbsolute) =>
        Path.IsPathRooted(relativeOrAbsolute) ? relativeOrAbsolute : Path.GetFullPath(Path.Combine(projectFolder, relativeOrAbsolute));

    /// <summary>Project-relative path with forward slashes when inside the project folder, otherwise absolute.</summary>
    public static string MakeRelative(string projectFolder, string fullPath)
    {
        var rel = Path.GetRelativePath(projectFolder, fullPath);
        return rel.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(rel) ? Path.GetFullPath(fullPath) : rel.Replace('\\', '/');
    }
}
