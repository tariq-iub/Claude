using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;
using CuAnnotator.Domain.Validation;

namespace CuAnnotator.Desktop.Views;

/// <summary>
/// Taxonomy editor: add/edit classes (name, ID, colour, formula, description, shortcut, parent category,
/// required verification, enabled). IDs already used by annotations cannot be removed or renumbered and the
/// reserved IDs 0/255 cannot be changed; validation runs before the edit is accepted.
/// </summary>
public sealed class TaxonomyEditorWindow : Window
{
    private readonly ObservableCollection<CorrosionClass> _rows;
    private readonly Taxonomy _original;
    private readonly IReadOnlyCollection<int> _used;
    private readonly TextBlock _error = new() { Foreground = System.Windows.Media.Brushes.Firebrick, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(12, 0, 12, 6) };

    public Taxonomy? Result { get; private set; }

    public TaxonomyEditorWindow(Taxonomy taxonomy, IReadOnlyCollection<int> usedIds)
    {
        _original = taxonomy; _used = usedIds;
        _rows = new ObservableCollection<CorrosionClass>(taxonomy.Clone().Classes.OrderBy(c => c.Id));
        Title = $"Taxonomy editor — {taxonomy.Name} (v{taxonomy.Version})";
        Width = 1100; Height = 560;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        SetResourceReference(StyleProperty, "AppWindow");
        Owner = System.Windows.Application.Current.MainWindow;

        var grid = new DataGrid { ItemsSource = _rows, CanUserAddRows = false, Margin = new Thickness(12), SelectionMode = DataGridSelectionMode.Single };
        grid.Columns.Add(new DataGridTextColumn { Header = "ID", Binding = new Binding(nameof(CorrosionClass.Id)), Width = 50 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Name", Binding = new Binding(nameof(CorrosionClass.Name)), Width = 170 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Formula", Binding = new Binding(nameof(CorrosionClass.ChemicalFormula)), Width = 110 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Colour (#RRGGBB)", Binding = new Binding(nameof(CorrosionClass.ColorHex)), Width = 110 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Key", Binding = new Binding(nameof(CorrosionClass.Shortcut)), Width = 45 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Parent category", Binding = new Binding(nameof(CorrosionClass.ParentCategory)), Width = 200 });
        grid.Columns.Add(new DataGridComboBoxColumn
        {
            Header = "Required verification", ItemsSource = Enum.GetValues<VerificationStatus>(),
            SelectedItemBinding = new Binding(nameof(CorrosionClass.RequiredVerification)), Width = 150,
        });
        grid.Columns.Add(new DataGridCheckBoxColumn { Header = "Enabled", Binding = new Binding(nameof(CorrosionClass.IsEnabled)), Width = 60 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Description", Binding = new Binding(nameof(CorrosionClass.Description)), Width = new DataGridLength(1, DataGridLengthUnitType.Star) });
        grid.BeginningEdit += (_, e) =>
        {
            if (e.Row.Item is CorrosionClass c && c.IsReserved && (string)e.Column.Header is "ID" or "Name")
            { e.Cancel = true; _error.Text = "Reserved classes (0 Unlabeled, 255 Ignore) keep their ID and name."; }
            else if (e.Row.Item is CorrosionClass u && _used.Contains(u.Id) && (string)e.Column.Header == "ID")
            { e.Cancel = true; _error.Text = $"ID {u.Id} is used by existing annotations and cannot be renumbered."; }
        };

        var add = Btn("Add class", "SecondaryButton", (_, _) =>
        {
            int id = Enumerable.Range(1, 254).First(i => _rows.All(r => r.Id != i));
            _rows.Add(new CorrosionClass { Id = id, Name = $"Class {id}", ColorHex = "#808080", RequiredVerification = VerificationStatus.ExpertReviewed });
        });
        var remove = Btn("Remove selected", "SecondaryButton", (_, _) =>
        {
            if (grid.SelectedItem is not CorrosionClass c) return;
            if (c.IsReserved || _used.Contains(c.Id)) { _error.Text = $"Class {c.Id} is reserved or in use; disable it instead of removing it."; return; }
            _rows.Remove(c);
        });
        var ok = Btn("Apply", "PrimaryButton", (_, _) => Apply());
        ok.IsDefault = true;
        var cancel = Btn("Cancel", "SecondaryButton", (_, _) => DialogResult = false);
        cancel.IsCancel = true;

        var buttons = new DockPanel { Margin = new Thickness(12, 0, 12, 12), LastChildFill = false };
        buttons.Children.Add(add); buttons.Children.Add(remove);
        DockPanel.SetDock(ok, Dock.Right); DockPanel.SetDock(cancel, Dock.Right);
        buttons.Children.Add(ok); buttons.Children.Add(cancel);
        var note = new TextBlock
        {
            Text = "Class IDs are the values written into semantic masks. They are permanent once used. The default IDs 1–5 match the original five-class copper-corrosion dataset.",
            TextWrapping = TextWrapping.Wrap, Margin = new Thickness(12, 10, 12, 0),
        };
        note.SetResourceReference(TextBlock.ForegroundProperty, "TextSecondary");

        var dock = new DockPanel();
        DockPanel.SetDock(note, Dock.Top); dock.Children.Add(note);
        DockPanel.SetDock(buttons, Dock.Bottom); dock.Children.Add(buttons);
        DockPanel.SetDock(_error, Dock.Bottom); dock.Children.Add(_error);
        dock.Children.Add(grid);
        Content = dock;
    }

    private static Button Btn(string text, string style, RoutedEventHandler click)
    {
        var b = new Button { Content = text, MinWidth = 90 };
        b.SetResourceReference(StyleProperty, style);
        b.Click += click;
        return b;
    }

    private void Apply()
    {
        var t = new Taxonomy { Name = _original.Name, Version = _original.Version, Classes = _rows.Select(c => c.Clone()).ToList() };
        var errors = TaxonomyValidator.ValidateEdit(_original, t, _used);
        if (errors.Count > 0) { _error.Text = string.Join("\n", errors); return; }
        Result = t;
        DialogResult = true;
    }
}
