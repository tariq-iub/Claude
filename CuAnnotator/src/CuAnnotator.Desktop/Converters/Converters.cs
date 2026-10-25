using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace CuAnnotator.Desktop.Converters;

/// <summary>Binds a RadioButton's IsChecked to an enum value (ConverterParameter = enum member name).</summary>
public sealed class EnumEqualsConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value?.ToString() == parameter?.ToString();

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is true && parameter is string s ? Enum.Parse(targetType, s) : Binding.DoNothing;
}

public sealed class NullToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        bool visible = value is not null && !(value is string s && s.Length == 0);
        return visible ^ Invert ? Visibility.Visible : Visibility.Collapsed;
    }
    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}

public sealed class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        (value is true) ^ Invert ? Visibility.Visible : Visibility.Collapsed;
    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}

/// <summary>Nullable double ↔ text (empty = null) for optional numeric fields.</summary>
public sealed class NullableDoubleConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is double d ? d.ToString("0.##", culture) : "";
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is string s && double.TryParse(s, NumberStyles.Float, culture, out var d) ? d : null;
}
