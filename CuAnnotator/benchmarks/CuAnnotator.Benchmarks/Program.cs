using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using CuAnnotator.Application.Annotations;
using CuAnnotator.Application.Export;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Geometry;
using CuAnnotator.Infrastructure.Persistence;
using CuAnnotator.Infrastructure.Serialization;
using CuAnnotator.Vision.ColorAnalysis;
using CuAnnotator.Vision.ImageIO;
using CuAnnotator.Vision.Segmentation;
using OpenCvSharp;

// Measured (not estimated) performance of the CPU pipeline. Each timing is the median of N runs after one
// warm-up. Usage: dotnet run -c Release --project benchmarks/CuAnnotator.Benchmarks -- [outputDir] [--quick]

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var outDir = Path.GetFullPath(args.FirstOrDefault(a => !a.StartsWith("--")) ?? Path.Combine("benchmarks", "results"));
bool quick = args.Contains("--quick");
int runs = quick ? 3 : 5;
Directory.CreateDirectory(outDir);
var tmp = Path.Combine(Path.GetTempPath(), "cuann-bench-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(tmp);
var proc = Process.GetCurrentProcess();
var rows = new ObservableList();

double Median(Func<double> f)
{
    f(); // warm-up
    var t = Enumerable.Range(0, runs).Select(_ => f()).OrderBy(x => x).ToList();
    return t[t.Count / 2];
}

double Time(Action a) { var sw = Stopwatch.StartNew(); a(); return sw.Elapsed.TotalMilliseconds; }
double TimeAsync(Func<Task> a) { var sw = Stopwatch.StartNew(); a().GetAwaiter().GetResult(); return sw.Elapsed.TotalMilliseconds; }

Mat Scene(int w, int h)
{
    var m = new Mat(h, w, MatType.CV_8UC3, new Scalar(70, 120, 190));
    var rnd = new Random(1);
    for (int i = 0; i < 40; i++)
        Cv2.Ellipse(m, new Point(rnd.Next(w), rnd.Next(h)), new Size(rnd.Next(20, w / 12), rnd.Next(20, h / 12)), rnd.Next(180), 0, 360,
            i % 3 == 0 ? new Scalar(40, 40, 150) : i % 3 == 1 ? new Scalar(30, 30, 35) : new Scalar(95, 150, 55), -1);
    Cv2.SetTheRNG(5);
    using var noise = new Mat(m.Size(), MatType.CV_16SC3);
    Cv2.Randn(noise, Scalar.All(0), Scalar.All(4));
    using var t = new Mat();
    m.ConvertTo(t, MatType.CV_16SC3); Cv2.Add(t, noise, t); t.ConvertTo(m, MatType.CV_8UC3);
    return m;
}

var engine = new SegmentationEngine();
foreach (var (name, w, h) in new[] { ("4K", 3840, 2160), ("8K", 7680, 4320) })
{
    Console.WriteLine($"── {name} {w}×{h}");
    using var img = Scene(w, h);
    var png = Path.Combine(tmp, $"{name}.png");
    Cv2.ImWrite(png, img);
    var size = $"{w}×{h}";
    rows.Add(("I/O", "Decode PNG + checksum + metadata", size, Median(() => Time(() => { using var l = ImageLoader.Load(png); })), $"{new FileInfo(png).Length / 1e6:0.0} MB file"));
    rows.Add(("Colour", "sRGB→CIELAB (exact, full image)", size, Median(() => Time(() => LabImage.FromBgr(img, new PixelRect(0, 0, w, h)))), $"{(long)w * h / 1e6:0.0} MP"));
    var labRoi = LabImage.FromBgr(img, new PixelRect(w / 4, h / 4, 1024, 1024));
    rows.Add(("Colour", "CIEDE2000 distance map 1024×1024", size, Median(() => Time(() => labRoi.DistanceMap(new LabColor(40, 30, 20), ColorDifferenceMetric.CIEDE2000))), "1.05 MP, parallel"));

    // GrabCut cost grows super-linearly with frame size; full-resolution 2048² ROIs take minutes and are omitted.
    var gcCases = new List<(int Roi, ProcessingQuality Q)> { (512, ProcessingQuality.Precise), (1024, ProcessingQuality.Balanced), (2048, ProcessingQuality.Balanced), (2048, ProcessingQuality.Fast) };
    if (name == "4K") gcCases.Add((1024, ProcessingQuality.Precise));
    foreach (var (roiSize, q) in gcCases)
    {
        var roi = new PixelRect(w / 2 - roiSize / 2, h / 2 - roiSize / 2, roiSize, roiSize);
        var p = new SegmentationParameters { Algorithm = SegmentationAlgorithmKind.GrabCut, Quality = q };
        rows.Add(("Segmentation", $"GrabCut 5 it. ({q})", $"ROI {roiSize}²", Median(() => TimeAsync(() => engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = roi, Parameters = p }, default))), $"{size} image"));
    }
    if (name == "4K")
    {
        var roi = new PixelRect(w / 2 - 400, h / 2 - 400, 800, 800);
        foreach (var k in new[] { SegmentationAlgorithmKind.LabColor, SegmentationAlgorithmKind.Watershed, SegmentationAlgorithmKind.ActiveContour, SegmentationAlgorithmKind.Hybrid })
        {
            var p = new SegmentationParameters { Algorithm = k };
            rows.Add(("Segmentation", k.ToString(), "ROI 800²", Median(() => TimeAsync(() => engine.SegmentAsync(new SegmentationRequest { Image = img, Roi = roi, Parameters = p }, default))), "default parameters"));
        }
    }

    // Project with many annotations: compose, export, save, load.
    var project = new AnnotationProject { Name = "bench" };
    var item = new ImageItem { Width = w, Height = h, FileName = $"{name}.png", RelativePath = png, ImportedAt = DateTimeOffset.UtcNow };
    project.Images.Add(item);
    var rr = new Random(3);
    for (int i = 0; i < 200; i++)
    {
        var s = rr.Next(40, 400);
        var bb = new PixelRect(rr.Next(w - s), rr.Next(h - s), s, s);
        var d = new byte[s * s];
        for (int y = 0; y < s; y++) for (int x = 0; x < s; x++) d[y * s + x] = (byte)((x - s / 2) * (x - s / 2) + (y - s / 2) * (y - s / 2) < s * s / 4 ? 1 : 0);
        var a = new Annotation { ImageId = item.Id, ClassId = 1 + i % 8, ZOrder = i, CreatedAt = DateTimeOffset.UtcNow };
        AnnotationGeometryService.SetMask(a, new RasterMask(bb, d));
        item.Annotations.Add(a);
    }
    rows.Add(("Masks", "Compose semantic label map (200 annotations)", size, Median(() => Time(() => LabelMapComposer.Compose(item))), ""));
    var map = LabelMapComposer.Compose(item);
    var maskPath = Path.Combine(tmp, "mask.png");
    rows.Add(("Masks", "Write 8-bit class-ID PNG", size, Median(() => Time(() => MaskIO.WriteLabelPng(maskPath, map.Labels, w, h))), ""));
    rows.Add(("Export", "COCO build + serialise (200 ann., auto mode)", size, Median(() => Time(() =>
        JsonSerializer.Serialize(CocoExporter.Build(project, new[] { (item, item.FileName) }, CocoSegmentationMode.Auto, new ExportReport())))), ""));
    rows.Add(("Export", "COCO build + serialise (200 ann., RLE)", size, Median(() => Time(() =>
        JsonSerializer.Serialize(CocoExporter.Build(project, new[] { (item, item.FileName) }, CocoSegmentationMode.AlwaysRle, new ExportReport())))), ""));
    var store = new ProjectStore(new NativeJsonSerializer());
    var pf = Path.Combine(tmp, "proj-" + name);
    rows.Add(("Project", "Save (JSON + SQLite index, 200 ann.)", size, Median(() => TimeAsync(() => store.SaveAsync(project, pf))), ""));
    rows.Add(("Project", "Load (200 ann.)", size, Median(() => TimeAsync(() => store.LoadAsync(pf))), $"{new FileInfo(Path.Combine(pf, "project.cuann.json")).Length / 1e6:0.00} MB JSON"));
    GC.Collect();
}

proc.Refresh();
var cpuName = RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && File.Exists("/proc/cpuinfo")
    ? File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name"))?.Split(':', 2)[1].Trim() ?? "unknown"
    : Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
var sb = new StringBuilder();
sb.AppendLine("# CuAnnotator benchmark results").AppendLine();
sb.AppendLine($"- Date (UTC): {DateTime.UtcNow:yyyy-MM-dd HH:mm}");
sb.AppendLine($"- Machine: {Environment.MachineName}; OS: {RuntimeInformation.OSDescription}; .NET {Environment.Version}");
sb.AppendLine($"- CPU: {cpuName}; logical processors visible: {Environment.ProcessorCount}");
#if DEBUG
const string buildConfig = "Debug (timings not representative)";
#else
const string buildConfig = "Release";
#endif
sb.AppendLine($"- OpenCV: {Cv2.GetVersionString()}; build configuration: {buildConfig}");
sb.AppendLine($"- Peak working set of the benchmark process: {proc.PeakWorkingSet64 / 1e6:0} MB");
sb.AppendLine($"- Total CPU time / wall time: {proc.TotalProcessorTime.TotalSeconds:0.0} s / {(DateTime.Now - proc.StartTime).TotalSeconds:0.0} s");
sb.AppendLine($"- Each value is the median of {runs} runs after one warm-up run. Synthetic images (ellipses + Gaussian noise).").AppendLine();
sb.AppendLine("| Group | Case | Size | Median (ms) | Note |").AppendLine("|---|---|---|---:|---|");
foreach (var r in rows) sb.AppendLine($"| {r.Group} | {r.Case} | {r.Size} | {r.Ms:0.0} | {r.Note} |");
sb.AppendLine().AppendLine("WPF rendering is not measured here (requires Windows); see documentation/11_Performance_Optimization.md.");
var file = Path.Combine(outDir, $"benchmark-{DateTime.UtcNow:yyyyMMdd}-{Environment.MachineName}.md");
File.WriteAllText(file, sb.ToString());
Console.WriteLine($"\nResults written to {file}");
Directory.Delete(tmp, true);

/// <summary>Row list that prints each measurement as soon as it is recorded.</summary>
sealed class ObservableList : List<(string Group, string Case, string Size, double Ms, string Note)>
{
    public new void Add((string Group, string Case, string Size, double Ms, string Note) r)
    {
        base.Add(r);
        Console.WriteLine($"  {r.Case,-48} {r.Size,-12} {r.Ms,10:0.0} ms  {r.Note}");
    }
}
