using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace CuAnnotator.Desktop.Views;

/// <summary>Read-only report window (quality reports, audit trail, help) with optional CSV/text export buttons.</summary>
public sealed class TextReportWindow : Window
{
    public TextReportWindow(string title, string text, IReadOnlyDictionary<string, Func<string>>? exports)
    {
        Title = title;
        Width = 900; Height = 640;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        SetResourceReference(StyleProperty, "AppWindow");
        Owner = System.Windows.Application.Current.MainWindow is { IsVisible: true } mw ? mw : null;

        var box = new TextBox
        {
            Text = text, IsReadOnly = true, FontFamily = new FontFamily("Cascadia Mono, Consolas"), FontSize = 12.5,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            TextWrapping = TextWrapping.NoWrap, Margin = new Thickness(12), BorderThickness = new Thickness(1),
        };
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(12, 0, 12, 12) };
        void Add(string label, Func<string> content, string filter, string ext)
        {
            var b = new Button { Content = label };
            b.SetResourceReference(StyleProperty, "SecondaryButton");
            b.Click += (_, _) =>
            {
                var d = new Microsoft.Win32.SaveFileDialog { Filter = filter, FileName = SafeName(label) + ext };
                if (d.ShowDialog(this) == true) File.WriteAllText(d.FileName, content());
            };
            buttons.Children.Add(b);
        }
        if (exports is not null) foreach (var (label, f) in exports) Add("Save " + label + "…", f, "CSV (*.csv)|*.csv|All files|*.*", ".csv");
        Add("Save as text…", () => text, "Text (*.txt)|*.txt", ".txt");
        var close = new Button { Content = "Close", IsCancel = true, MinWidth = 80 };
        close.SetResourceReference(StyleProperty, "PrimaryButton");
        close.Click += (_, _) => Close();
        buttons.Children.Add(close);

        var dock = new DockPanel();
        DockPanel.SetDock(buttons, Dock.Bottom);
        dock.Children.Add(buttons);
        dock.Children.Add(box);
        Content = dock;
    }

    private static string SafeName(string s) => string.Concat(s.Where(c => char.IsLetterOrDigit(c) || c == ' ')).Trim().Replace(' ', '_').ToLowerInvariant();
}
