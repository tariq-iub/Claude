using System.Windows;

namespace CuAnnotator.Desktop.Services;

public sealed class ThemeService
{
    public string Current { get; private set; } = "Light";

    public void Apply(string theme)
    {
        theme = theme == "Dark" ? "Dark" : "Light";
        var dicts = System.Windows.Application.Current.Resources.MergedDictionaries;
        var replacement = new ResourceDictionary { Source = new Uri($"pack://application:,,,/CuAnnotator;component/Themes/{theme}.xaml") };
        if (dicts.Count > 0) dicts[0] = replacement; else dicts.Insert(0, replacement);
        Current = theme;
    }
}
