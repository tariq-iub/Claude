using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using CuAnnotator.Desktop.Interaction;
using CuAnnotator.Desktop.Services;
using CuAnnotator.Desktop.ViewModels;
using CuAnnotator.Domain.Entities;
using CuAnnotator.Domain.Enums;

namespace CuAnnotator.Desktop.Views;

/// <summary>
/// View wiring only: keyboard routing, viewport/minimap hookup and panel layout. All behaviour lives in the
/// view models and the application layer.
/// </summary>
public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;
    private readonly ToolController _tools;
    private bool _closingConfirmed;
    private GridLength _lastBrowser = new(132), _lastPanel = new(340);

    public MainWindow(MainViewModel vm)
    {
        InitializeComponent();
        _vm = vm;
        DataContext = vm;
        _tools = new ToolController(Viewport, vm);
        Minimap.Attach(Viewport);

        vm.ViewCommandRequested += (_, c) =>
        {
            switch (c)
            {
                case ViewCommand.ZoomIn: Viewport.ZoomBy(1.25); break;
                case ViewCommand.ZoomOut: Viewport.ZoomBy(0.8); break;
                case ViewCommand.Fit: Viewport.FitToWindow(); break;
                case ViewCommand.ActualPixels: Viewport.ActualPixels(); break;
            }
        };
        vm.ViewCenterRequested += (_, p) => Dispatcher.BeginInvoke(() => Viewport.CenterOn(p));
        Viewport.TransformChanged += (_, _) => vm.ZoomText = $"Zoom {Viewport.ZoomPercent:0.#} %";
        vm.PropertyChanged += OnViewModelPropertyChanged;
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(MainViewModel.Editor)) return;
        if (_vm.Editor is { } ed)
        {
            Minimap.SetImage(ed.Tiles.Thumbnail(400), ed.Image.Width, ed.Image.Height);
            ed.PropertyChanged += (_, pe) =>
            {
                if (pe.PropertyName == nameof(EditorViewModel.SelectedAnnotation) && _vm.ActiveTool == ToolKind.VertexEditor) _tools.BeginVertexEditing();
            };
        }
        else Minimap.SetImage(null, 0, 0);
    }

    // ───────────── Keyboard ─────────────

    protected override void OnPreviewKeyDown(KeyEventArgs e)
    {
        base.OnPreviewKeyDown(e);
        var key = e.Key == Key.System ? e.SystemKey : e.Key;
        bool inText = Keyboard.FocusedElement is TextBox or ComboBox || (Keyboard.FocusedElement as FrameworkElement)?.TemplatedParent is ComboBox;
        var mods = Keyboard.Modifiers;

        if (key == Key.Space && !inText) { Viewport.SpaceHeld = true; e.Handled = true; return; }
        if (key == Key.F1) { _vm.HelpCommand.Execute(null); e.Handled = true; return; }
        // In text fields only Ctrl+S is global; everything else types normally.
        if (inText && !(mods == ModifierKeys.Control && key == Key.S)) return;

        var action = KeyMapService.Resolve(_vm.Settings.KeyMap, key, mods);
        if (action is not null && Execute(action)) { e.Handled = true; return; }

        // Class shortcuts (1–8 by default, configurable in the taxonomy editor).
        if (mods == ModifierKeys.None && ClassForKey(key) is { } cls)
        {
            _vm.AssignClassCommand.Execute(cls.Id);
            e.Handled = true;
        }
    }

    protected override void OnPreviewKeyUp(KeyEventArgs e)
    {
        base.OnPreviewKeyUp(e);
        if (e.Key == Key.Space) Viewport.SpaceHeld = false;
    }

    private CorrosionClass? ClassForKey(Key key)
    {
        string? text = key switch
        {
            >= Key.D0 and <= Key.D9 => ((int)(key - Key.D0)).ToString(),
            >= Key.NumPad0 and <= Key.NumPad9 => ((int)(key - Key.NumPad0)).ToString(),
            >= Key.A and <= Key.Z => key.ToString(),
            _ => null,
        };
        if (text is null || _vm.Project is null) return null;
        return _vm.Project.Taxonomy.Classes.FirstOrDefault(c => c.IsEnabled && string.Equals(c.Shortcut, text, StringComparison.OrdinalIgnoreCase));
    }

    private bool Execute(string action)
    {
        if (KeyMapService.ToolFor(action) is { } tool) { _vm.ActiveTool = tool; return true; }
        var ed = _vm.Editor;
        switch (action)
        {
            case "OpenProject": _vm.OpenProjectCommand.Execute(null); return true;
            case "SaveProject": _vm.SaveProjectCommand.Execute(null); return true;
            case "Undo": if (_vm.UndoCommand.CanExecute(null)) _vm.UndoCommand.Execute(null); return true;
            case "Redo": if (_vm.RedoCommand.CanExecute(null)) _vm.RedoCommand.Execute(null); return true;
            case "RunGrabCut":
                if (ed?.Session.Candidate is null) return false;
                ed.RunSegmentationCommand.Execute(null); return true;
            case "FitImage": Viewport.FitToWindow(); return true;
            case "ActualPixels": Viewport.ActualPixels(); return true;
            case "ZoomIn": Viewport.ZoomBy(1.25); return true;
            case "ZoomOut": Viewport.ZoomBy(0.8); return true;
            case "ToggleOverlays": _vm.ToggleOverlaysCommand.Execute(null); return true;
            case "Delete": if (ed?.DeleteSelectedCommand.CanExecute(null) == true) ed.DeleteSelectedCommand.Execute(null); return true;
            case "Accept": return _tools.OnEnter();
            case "Cancel": return _tools.OnEscape();
            case "NextImage": _vm.NextImageCommand.Execute(null); return true;
            case "PreviousImage": _vm.PreviousImageCommand.Execute(null); return true;
            case "AssignIgnore": _vm.AssignClassCommand.Execute(Taxonomy.IgnoreId); return true;
        }
        return false;
    }

    // ───────────── Panels ─────────────

    private void TogglePanel(object sender, MouseButtonEventArgs e)
    {
        if (PanelColumn.Width.Value > 0) { _lastPanel = PanelColumn.Width; PanelColumn.Width = new GridLength(0); }
        else PanelColumn.Width = _lastPanel;
    }

    private void ToggleBrowser(object sender, MouseButtonEventArgs e)
    {
        if (BrowserRow.Height.Value > 0) { _lastBrowser = BrowserRow.Height; BrowserRow.Height = new GridLength(0); }
        else BrowserRow.Height = _lastBrowser;
    }

    private void ClassDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (sender is ListBoxItem { DataContext: ClassItemViewModel c }) _vm.AssignClassCommand.Execute(c.Id);
    }

    private void WorkflowChanged(object sender, SelectionChangedEventArgs e)
    {
        if (sender is ComboBox { IsDropDownOpen: true } or ComboBox { IsKeyboardFocusWithin: true } && e.AddedItems.Count == 1
            && e.AddedItems[0] is ImageWorkflowState s && _vm.SelectedImage?.State != s)
            _vm.SetWorkflowStateCommand.Execute(s);
    }

    protected override async void OnClosing(CancelEventArgs e)
    {
        base.OnClosing(e);
        if (_closingConfirmed) return;
        e.Cancel = true;
        if (!await _vm.ConfirmDiscardAsync()) return;
        _closingConfirmed = true;
        _vm.Editor?.Dispose();
        Close();
    }
}
