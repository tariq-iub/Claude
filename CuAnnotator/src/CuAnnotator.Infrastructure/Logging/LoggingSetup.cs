using Serilog;

namespace CuAnnotator.Infrastructure.Logging;

public static class LoggingSetup
{
    /// <summary>Default log folder: %LOCALAPPDATA%/CuAnnotator/logs (rolling daily files, 14 retained).</summary>
    public static string DefaultLogFolder =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CuAnnotator", "logs");

    public static ILogger Create(string? folder = null)
    {
        folder ??= DefaultLogFolder;
        Directory.CreateDirectory(folder);
        return new LoggerConfiguration()
            .MinimumLevel.Debug()
            .Enrich.WithProperty("App", "CuAnnotator")
            .WriteTo.File(Path.Combine(folder, "cuannotator-.log"), rollingInterval: RollingInterval.Day, retainedFileCountLimit: 14,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();
    }
}
