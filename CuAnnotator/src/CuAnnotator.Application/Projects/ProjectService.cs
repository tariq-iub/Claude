using CuAnnotator.Application.Services;
using CuAnnotator.Contracts;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Validation;
using CuAnnotator.Vision.ImageIO;

namespace CuAnnotator.Application.Projects;

public sealed class NewProjectRequest
{
    public required string Name { get; init; }
    public required string Folder { get; init; }
    public string Description { get; init; } = "";
    public string Researcher { get; init; } = "";
    public string Institution { get; init; } = "";
    public Taxonomy? Taxonomy { get; init; }
    public ProjectSettings? Settings { get; init; }
}

public sealed record ImportResult(IReadOnlyList<ImageItem> Imported, IReadOnlyList<string> Skipped);

/// <summary>Project lifecycle: create, open, save, import images.</summary>
public sealed class ProjectService(IProjectStore store, TimeProvider? clock = null)
{
    private readonly TimeProvider _clock = clock ?? TimeProvider.System;

    public AnnotationProject Create(NewProjectRequest r)
    {
        if (string.IsNullOrWhiteSpace(r.Name)) throw new ArgumentException("Project name is required.");
        Directory.CreateDirectory(r.Folder);
        if (File.Exists(Path.Combine(r.Folder, NativeFormat.ProjectFileName)))
            throw new InvalidOperationException($"A project already exists in '{r.Folder}'.");
        var now = _clock.GetUtcNow();
        var p = new AnnotationProject
        {
            Name = r.Name, Description = r.Description, Researcher = r.Researcher, Institution = r.Institution,
            CreatedAt = now, ModifiedAt = now, CreatedWithAppVersion = AppInfo.Version,
            Taxonomy = r.Taxonomy?.Clone() ?? Taxonomy.CreateDefault(),
            Settings = r.Settings ?? new ProjectSettings(),
        };
        var errors = TaxonomyValidator.Validate(p.Taxonomy);
        if (errors.Count > 0) throw new InvalidOperationException("Invalid taxonomy: " + string.Join(" ", errors));
        Directory.CreateDirectory(Path.Combine(r.Folder, NativeFormat.ImagesFolder));
        return p;
    }

    public Task SaveAsync(AnnotationProject p, string folder, CancellationToken ct = default)
    {
        p.ModifiedAt = _clock.GetUtcNow();
        p.SavedWithAppVersion = AppInfo.Version;
        return store.SaveAsync(p, folder, ct);
    }

    public Task<AnnotationProject> OpenAsync(string folder, CancellationToken ct = default) => store.LoadAsync(folder, ct);

    /// <summary>
    /// Imports images: decodes each file to record dimensions/format/orientation/checksum and, if the
    /// project setting asks for it, copies the original (byte-identical) into the project's images folder.
    /// Duplicates (same SHA-256) are skipped.
    /// </summary>
    public async Task<ImportResult> ImportImagesAsync(AnnotationProject project, string projectFolder, IEnumerable<string> files,
        string? specimenId = null, string? acquisitionSource = null, IProgress<(int, int, string)>? progress = null, CancellationToken ct = default)
    {
        var list = files.ToList();
        var imported = new List<ImageItem>(); var skipped = new List<string>();
        var known = project.Images.Select(i => i.Sha256).ToHashSet();
        int n = 0;
        foreach (var f in list)
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report((n++, list.Count, Path.GetFileName(f)));
            if (!ImageLoader.IsSupported(f)) { skipped.Add($"{f}: unsupported format"); continue; }
            ImageInfo info;
            try
            {
                info = await Task.Run(() => { using var li = ImageLoader.Load(f); return li.Info; }, ct);
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or OpenCvSharp.OpenCVException)
            {
                skipped.Add($"{f}: {ex.Message}"); continue;
            }
            if (!known.Add(info.Sha256)) { skipped.Add($"{f}: duplicate of an image already in the project"); continue; }

            string stored = Path.GetFullPath(f);
            if (project.Settings.CopyImagesIntoProject)
            {
                var dir = Path.Combine(projectFolder, NativeFormat.ImagesFolder);
                Directory.CreateDirectory(dir);
                var target = Path.Combine(dir, Path.GetFileName(f));
                int k = 2;
                while (File.Exists(target) && ImageLoader.ComputeSha256(target) != info.Sha256)
                    target = Path.Combine(dir, $"{Path.GetFileNameWithoutExtension(f)}_{k++}{Path.GetExtension(f)}");
                if (!File.Exists(target)) File.Copy(f, target);
                stored = target;
            }
            var item = new ImageItem
            {
                FileName = Path.GetFileName(f),
                RelativePath = ProjectPaths.MakeRelative(projectFolder, stored),
                OriginalSourcePath = Path.GetFullPath(f),
                Width = info.Width, Height = info.Height, PixelFormat = info.PixelFormat, BitDepth = info.BitDepth, Channels = info.Channels,
                FileSizeBytes = info.FileSizeBytes, Sha256 = info.Sha256, ImportedAt = _clock.GetUtcNow(),
                ColorProfile = info.ColorProfile, ExifOrientation = info.ExifOrientation, ExifOrientationApplied = info.ExifOrientationApplied,
                SpecimenId = specimenId, AcquisitionSource = acquisitionSource,
                Notes = string.Join(" ", info.ConversionNotes),
            };
            project.Images.Add(item);
            imported.Add(item);
        }
        progress?.Report((list.Count, list.Count, ""));
        return new ImportResult(imported, skipped);
    }

    /// <summary>Loads an image of the project and verifies it against the recorded checksum and size.</summary>
    public static LoadedImage LoadImage(string projectFolder, ImageItem item, out string? warning)
    {
        var path = ProjectPaths.Resolve(projectFolder, item.RelativePath);
        var li = ImageLoader.Load(path);
        warning = null;
        if (li.Info.Width != item.Width || li.Info.Height != item.Height)
        {
            li.Dispose();
            throw new InvalidDataException($"'{item.FileName}' is now {li.Info.Width}×{li.Info.Height}, but annotations were made on {item.Width}×{item.Height}.");
        }
        if (!string.IsNullOrEmpty(item.Sha256) && li.Info.Sha256 != item.Sha256)
            warning = $"'{item.FileName}' changed on disk since import (checksum mismatch). Annotations may no longer correspond to the pixels.";
        return li;
    }
}
