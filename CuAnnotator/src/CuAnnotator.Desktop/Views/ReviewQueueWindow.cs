using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Desktop.Views;

public sealed record ReviewRow(Guid ImageId, Guid AnnotationId, string Image, string Class, VerificationStatus Verification, long Area,
    string CreatedBy, string Reviewer, DateTimeOffset Modified, ImageWorkflowState ImageState, string Set, int Revision);

/// <summary>Annotation review queue with filters (image, class, verification, reviewer, date, completion state).</summary>
public sealed class ReviewQueueWindow : Window
{
    private readonly List<ReviewRow> _all;
    private readonly DataGrid _grid = new() { IsReadOnly = true, Margin = new Thickness(12, 0, 12, 8) };
    private readonly ComboBox _class = new() { Width = 170, Margin = new Thickness(0, 0, 8, 0) };
    private readonly ComboBox _verif = new() { Width = 170, Margin = new Thickness(0, 0, 8, 0) };
    private readonly ComboBox _state = new() { Width = 140, Margin = new Thickness(0, 0, 8, 0) };
    private readonly TextBox _text = new() { Width = 180, Margin = new Thickness(0, 0, 8, 0), ToolTip = "Filter by image name, reviewer or author" };
    private readonly DatePicker _since = new() { Width = 130, ToolTip = "Modified on or after" };
    private readonly TextBlock _count = new() { VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(8, 0, 0, 0) };

    public (Guid ImageId, Guid AnnotationId)? Result { get; private set; }

    public ReviewQueueWindow(AnnotationProject project)
    {
        Title = "Annotation review queue";
        Width = 1150; Height = 620;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        SetResourceReference(StyleProperty, "AppWindow");
        Owner = System.Windows.Application.Current.MainWindow;

        _all = project.Images.SelectMany(i => i.Annotations.Select(a => new ReviewRow(i.Id, a.Id, i.FileName,
            $"{a.ClassId} {project.Taxonomy.Find(a.ClassId)?.Name}", a.VerificationStatus, a.PixelArea, a.CreatedBy, a.ReviewedBy ?? "",
            a.ModifiedAt, i.WorkflowState, a.AnnotationSet, a.Revision))).OrderBy(r => r.Verification).ThenBy(r => r.Image).ToList();

        _class.ItemsSource = new[] { "(all classes)" }.Concat(project.Taxonomy.Classes.Where(c => c.Id != 0).Select(c => $"{c.Id} {c.Name}")).ToList();
        _class.SelectedIndex = 0;
        _verif.ItemsSource = new object[] { "(any verification)" }.Concat(Enum.GetValues<VerificationStatus>().Cast<object>()).ToList();
        _verif.SelectedIndex = 0;
        _state.ItemsSource = new object[] { "(any image state)" }.Concat(Enum.GetValues<ImageWorkflowState>().Cast<object>()).ToList();
        _state.SelectedIndex = 0;
        _class.SelectionChanged += (_, _) => Apply();
        _verif.SelectionChanged += (_, _) => Apply();
        _state.SelectionChanged += (_, _) => Apply();
        _text.TextChanged += (_, _) => Apply();
        _since.SelectedDateChanged += (_, _) => Apply();

        foreach (var (h, p) in new[] { ("Image", "Image"), ("Class", "Class"), ("Verification", "Verification"), ("Area (px)", "Area"), ("Author", "CreatedBy"),
                     ("Reviewer", "Reviewer"), ("Modified", "Modified"), ("Image state", "ImageState"), ("Set", "Set"), ("Rev.", "Revision") })
            _grid.Columns.Add(new DataGridTextColumn { Header = h, Binding = new Binding(p) });
        _grid.MouseDoubleClick += (_, _) => Go();

        var filters = new WrapPanel { Margin = new Thickness(12, 12, 12, 8) };
        foreach (var c in new UIElement[] { _class, _verif, _state, _text, _since, _count }) filters.Children.Add(c);
        var open = new Button { Content = "Go to annotation", IsDefault = true };
        open.SetResourceReference(StyleProperty, "PrimaryButton");
        open.Click += (_, _) => Go();
        var close = new Button { Content = "Close", IsCancel = true };
        close.SetResourceReference(StyleProperty, "SecondaryButton");
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(12, 0, 12, 12) };
        buttons.Children.Add(close); buttons.Children.Add(open);

        var dock = new DockPanel();
        DockPanel.SetDock(filters, Dock.Top); dock.Children.Add(filters);
        DockPanel.SetDock(buttons, Dock.Bottom); dock.Children.Add(buttons);
        dock.Children.Add(_grid);
        Content = dock;
        Apply();
    }

    private void Apply()
    {
        IEnumerable<ReviewRow> q = _all;
        if (_class.SelectedIndex > 0) q = q.Where(r => r.Class == (string)_class.SelectedItem);
        if (_verif.SelectedItem is VerificationStatus v) q = q.Where(r => r.Verification == v);
        if (_state.SelectedItem is ImageWorkflowState s) q = q.Where(r => r.ImageState == s);
        if (!string.IsNullOrWhiteSpace(_text.Text))
            q = q.Where(r => r.Image.Contains(_text.Text, StringComparison.OrdinalIgnoreCase) || r.Reviewer.Contains(_text.Text, StringComparison.OrdinalIgnoreCase)
                             || r.CreatedBy.Contains(_text.Text, StringComparison.OrdinalIgnoreCase));
        if (_since.SelectedDate is { } d) q = q.Where(r => r.Modified.LocalDateTime.Date >= d.Date);
        var list = q.ToList();
        _grid.ItemsSource = list;
        _count.Text = $"{list.Count} of {_all.Count} annotations";
    }

    private void Go()
    {
        if (_grid.SelectedItem is not ReviewRow r) return;
        Result = (r.ImageId, r.AnnotationId);
        DialogResult = true;
    }
}
