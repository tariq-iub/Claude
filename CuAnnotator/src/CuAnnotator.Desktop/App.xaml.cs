using System.IO;
using System.Windows;
using System.Windows.Threading;
using CuAnnotator.Application.Export;
using CuAnnotator.Application.Projects;
using CuAnnotator.Application.Services;
using CuAnnotator.Desktop.Services;
using CuAnnotator.Desktop.ViewModels;
using CuAnnotator.Desktop.Views;
using CuAnnotator.Infrastructure.Logging;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Vision.Segmentation;
using Microsoft.Extensions.DependencyInjection;
using Serilog;

namespace CuAnnotator.Desktop;

public partial class App : System.Windows.Application
{
    public static IServiceProvider Services { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        var logger = LoggingSetup.Create();
        Log.Logger = logger;
        logger.Information("CuAnnotator {Version} starting on {OS}, {Cpu} logical CPUs", AppInfo.Version, Environment.OSVersion, Environment.ProcessorCount);

        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += (_, a) => logger.Fatal(a.ExceptionObject as Exception, "Unhandled exception");
        TaskScheduler.UnobservedTaskException += (_, a) => { logger.Error(a.Exception, "Unobserved task exception"); a.SetObserved(); };

        var settings = UserSettings.Load();
        var services = new ServiceCollection();
        services.AddSingleton<ILogger>(logger);
        services.AddSingleton(settings);
        services.AddSingleton<IProjectSerializer, NativeJsonSerializer>();
        services.AddSingleton<ProjectStore>(sp => new ProjectStore(sp.GetRequiredService<IProjectSerializer>(), logger));
        services.AddSingleton<IProjectStore>(sp => sp.GetRequiredService<ProjectStore>());
        services.AddSingleton<ProjectService>(sp => new ProjectService(sp.GetRequiredService<IProjectStore>()));
        services.AddSingleton<SegmentationEngine>();
        services.AddSingleton<DatasetExporter>();
        services.AddSingleton<ThemeService>();
        services.AddSingleton<IDialogService, DialogService>();
        services.AddSingleton<MainViewModel>();
        services.AddSingleton<MainWindow>();
        Services = services.BuildServiceProvider();

        Services.GetRequiredService<ThemeService>().Apply(settings.Theme);
        var window = Services.GetRequiredService<MainWindow>();
        MainWindow = window;
        window.Show();
        if (e.Args.Length > 0 && Directory.Exists(e.Args[0]))
            _ = Services.GetRequiredService<MainViewModel>().OpenProjectFolderAsync(e.Args[0]);
        else if (settings.ShowOnboarding)
            Services.GetRequiredService<IDialogService>().ShowOnboarding();
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        Log.Error(e.Exception, "Unhandled UI exception");
        MessageBox.Show(MainWindow!, $"An unexpected error occurred:\n\n{e.Exception.Message}\n\nYour project has not been modified on disk by this error. " +
                        "Details were written to the log folder.", "CuAnnotator", MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Log.Information("CuAnnotator exiting");
        Log.CloseAndFlush();
        base.OnExit(e);
    }
}
