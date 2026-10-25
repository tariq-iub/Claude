using System.IO;
using System.Text.Json;
using System.Windows.Input;
using CuAnnotator.Desktop.Interaction;

namespace CuAnnotator.Desktop.Services;

/// <summary>Per-user preferences stored in %APPDATA%/CuAnnotator/settings.json (not part of any project).</summary>
public sealed class UserSettings
{
    public string Theme { get; set; } = "Light";
    public string UserName { get; set; } = Environment.UserName;
    public bool ShowOnboarding { get; set; } = true;
    public List<string> RecentProjects { get; set; } = new();
    public double OverlayOpacity { get; set; } = 0.45;
    public int BrushRadius { get; set; } = 6;
    public int SnapRadius { get; set; } = 8;
    public double EditSimplification { get; set; } = 1.0;
    public bool LivePreview { get; set; } = true;
    /// <summary>Action name → key gesture string (e.g. "Ctrl+S", "R", "Shift+Tab").</summary>
    public Dictionary<string, string> KeyMap { get; set; } = KeyMapService.Defaults();

    public static string FilePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "CuAnnotator", "settings.json");

    public static UserSettings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                var s = JsonSerializer.Deserialize<UserSettings>(File.ReadAllText(FilePath)) ?? new UserSettings();
                foreach (var kv in KeyMapService.Defaults()) s.KeyMap.TryAdd(kv.Key, kv.Value); // new actions in upgrades
                return s;
            }
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            Serilog.Log.Warning(ex, "Could not read user settings; using defaults");
        }
        return new UserSettings();
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
            File.WriteAllText(FilePath, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Serilog.Log.Warning(ex, "Could not save user settings");
        }
    }

    public void AddRecent(string folder)
    {
        RecentProjects.RemoveAll(p => string.Equals(p, folder, StringComparison.OrdinalIgnoreCase));
        RecentProjects.Insert(0, folder);
        if (RecentProjects.Count > 10) RecentProjects.RemoveRange(10, RecentProjects.Count - 10);
        Save();
    }
}

/// <summary>Configurable keyboard shortcuts (section 20).</summary>
public static class KeyMapService
{
    public static Dictionary<string, string> Defaults() => new()
    {
        ["OpenProject"] = "Ctrl+O", ["SaveProject"] = "Ctrl+S", ["Undo"] = "Ctrl+Z", ["Redo"] = "Ctrl+Y",
        ["Tool.Select"] = "V", ["Tool.Rectangle"] = "R", ["Tool.Polygon"] = "P", ["Tool.Lasso"] = "L", ["Tool.SmartContour"] = "S",
        ["Tool.ForegroundBrush"] = "B", ["Tool.BackgroundBrush"] = "N", ["Tool.BoundaryBrush"] = "K", ["Tool.Eraser"] = "E",
        ["Tool.VertexEditor"] = "A", ["Tool.Split"] = "X", ["Tool.Merge"] = "M", ["Tool.Fill"] = "H", ["Tool.ColorSelect"] = "C",
        ["Tool.MagicWand"] = "W", ["Tool.Measure"] = "U",
        ["RunGrabCut"] = "G", ["FitImage"] = "F", ["ActualPixels"] = "Ctrl+D1", ["ZoomIn"] = "Ctrl+OemPlus", ["ZoomOut"] = "Ctrl+OemMinus",
        ["ToggleOverlays"] = "O", ["Delete"] = "Delete", ["Accept"] = "Enter", ["Cancel"] = "Escape",
        ["NextImage"] = "Tab", ["PreviousImage"] = "Shift+Tab", ["AssignIgnore"] = "I",
    };

    /// <summary>Parses "Ctrl+Shift+S", "Tab", "R", "Ctrl+OemPlus" into key and modifiers.</summary>
    public static bool TryParse(string gesture, out Key key, out ModifierKeys mods)
    {
        key = Key.None; mods = ModifierKeys.None;
        var parts = gesture.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0) return false;
        foreach (var m in parts[..^1])
        {
            switch (m.ToLowerInvariant())
            {
                case "ctrl": case "control": mods |= ModifierKeys.Control; break;
                case "shift": mods |= ModifierKeys.Shift; break;
                case "alt": mods |= ModifierKeys.Alt; break;
                default: return false;
            }
        }
        return Enum.TryParse(parts[^1], true, out key);
    }

    public static bool Matches(string gesture, Key key, ModifierKeys mods) =>
        TryParse(gesture, out var k, out var m) && k == key && m == mods;

    /// <summary>Returns the action bound to a key press, or null.</summary>
    public static string? Resolve(IReadOnlyDictionary<string, string> map, Key key, ModifierKeys mods) =>
        map.FirstOrDefault(kv => Matches(kv.Value, key, mods)).Key;

    public static ToolKind? ToolFor(string action) =>
        action.StartsWith("Tool.") && Enum.TryParse<ToolKind>(action[5..], out var t) ? t : null;
}
