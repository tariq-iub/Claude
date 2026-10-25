using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace CuAnnotator.Desktop.Views;

/// <summary>Small form dialog built from typed fields; used for the simple dialogs of the application.</summary>
public sealed class FormWindow : Window
{
    private readonly StackPanel _fields = new() { Margin = new Thickness(16, 8, 16, 8) };
    private readonly Dictionary<string, Func<object?>> _getters = new();
    private readonly List<Func<string?>> _validators = new();
    private readonly TextBlock _error = new() { Foreground = Brushes.Firebrick, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(16, 0, 16, 4) };

    public FormWindow(string title, string? intro = null, double width = 460)
    {
        Title = title;
        Width = width;
        SizeToContent = SizeToContent.Height;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        ResizeMode = ResizeMode.NoResize;
        ShowInTaskbar = false;
        SetResourceReference(StyleProperty, "AppWindow");
        Owner = System.Windows.Application.Current.MainWindow is { IsVisible: true } mw ? mw : null;

        var root = new DockPanel();
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(16, 4, 16, 14) };
        var ok = new Button { Content = "OK", IsDefault = true, MinWidth = 84 };
        ok.SetResourceReference(StyleProperty, "PrimaryButton");
        ok.Click += (_, _) =>
        {
            var problem = _validators.Select(v => v()).FirstOrDefault(m => m is not null);
            if (problem is not null) { _error.Text = problem; return; }
            DialogResult = true;
        };
        var cancel = new Button { Content = "Cancel", IsCancel = true, MinWidth = 84 };
        cancel.SetResourceReference(StyleProperty, "SecondaryButton");
        buttons.Children.Add(cancel); buttons.Children.Add(ok);
        DockPanel.SetDock(buttons, Dock.Bottom);
        root.Children.Add(buttons);
        DockPanel.SetDock(_error, Dock.Bottom);
        root.Children.Add(_error);
        if (intro is not null)
        {
            var t = new TextBlock { Text = intro, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(16, 14, 16, 2) };
            t.SetResourceReference(TextBlock.ForegroundProperty, "TextSecondary");
            DockPanel.SetDock(t, Dock.Top);
            root.Children.Add(t);
        }
        root.Children.Add(new ScrollViewer { Content = _fields, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, MaxHeight = 640 });
        Content = root;
    }

    private void AddLabel(string label)
    {
        var t = new TextBlock { Text = label, Margin = new Thickness(0, 8, 0, 2) };
        t.SetResourceReference(TextBlock.ForegroundProperty, "TextSecondary");
        _fields.Children.Add(t);
    }

    public FormWindow Text(string key, string label, string value = "", bool required = false, bool multiline = false)
    {
        AddLabel(label);
        var tb = new TextBox { Text = value, AcceptsReturn = multiline, TextWrapping = multiline ? TextWrapping.Wrap : TextWrapping.NoWrap, MinHeight = multiline ? 60 : 0 };
        _fields.Children.Add(tb);
        _getters[key] = () => tb.Text.Trim();
        if (required) _validators.Add(() => string.IsNullOrWhiteSpace(tb.Text) ? $"'{label}' is required." : null);
        return this;
    }

    public FormWindow Folder(string key, string label, string value = "")
    {
        AddLabel(label);
        var row = new DockPanel();
        var browse = new Button { Content = "Browse…", Margin = new Thickness(6, 0, 0, 0) };
        browse.SetResourceReference(StyleProperty, "SecondaryButton");
        var tb = new TextBox { Text = value };
        browse.Click += (_, _) =>
        {
            var d = new Microsoft.Win32.OpenFolderDialog { Title = label };
            if (d.ShowDialog(this) == true) tb.Text = d.FolderName;
        };
        DockPanel.SetDock(browse, Dock.Right);
        row.Children.Add(browse); row.Children.Add(tb);
        _fields.Children.Add(row);
        _getters[key] = () => tb.Text.Trim();
        _validators.Add(() => string.IsNullOrWhiteSpace(tb.Text) ? $"'{label}' is required." : null);
        return this;
    }

    public FormWindow Number(string key, string label, double? value, double min = double.MinValue, double max = double.MaxValue, bool optional = false)
    {
        AddLabel(label);
        var tb = new TextBox { Text = value?.ToString(CultureInfo.CurrentCulture) ?? "" };
        _fields.Children.Add(tb);
        _getters[key] = () => double.TryParse(tb.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out var v) ? v : (double?)null;
        _validators.Add(() =>
        {
            if (string.IsNullOrWhiteSpace(tb.Text)) return optional ? null : $"'{label}' is required.";
            return double.TryParse(tb.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out var v) && v >= min && v <= max
                ? null : $"'{label}' must be a number between {min} and {max}.";
        });
        return this;
    }

    public FormWindow Choice<T>(string key, string label, IEnumerable<T> items, T selected)
    {
        AddLabel(label);
        var cb = new ComboBox { ItemsSource = items.ToList(), SelectedItem = selected };
        _fields.Children.Add(cb);
        _getters[key] = () => cb.SelectedItem;
        return this;
    }

    public FormWindow Check(string key, string label, bool value)
    {
        var c = new CheckBox { Content = label, IsChecked = value, Margin = new Thickness(0, 6, 0, 0) };
        _fields.Children.Add(c);
        _getters[key] = () => c.IsChecked == true;
        return this;
    }

    public FormWindow Note(string text)
    {
        var t = new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 8, 0, 0), FontStyle = FontStyles.Italic };
        t.SetResourceReference(TextBlock.ForegroundProperty, "TextSecondary");
        _fields.Children.Add(t);
        return this;
    }

    public T Get<T>(string key) => (T)_getters[key]()!;
    public string GetText(string key) => (string)_getters[key]()!;
    public double? GetNumber(string key) => (double?)_getters[key]();
    public bool GetBool(string key) => (bool)_getters[key]()!;

    public bool Run() => ShowDialog() == true;
}
