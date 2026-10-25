# 07 — WPF UI Architecture

## Main window regions

| Region | Contents | Code |
|---|---|---|
| A. Toolbar | New, Open, Save, Save as, Import, Export, Undo, Redo, Zoom ±, Fit, 1:1, Overlays, algorithm selector, Classes, Review, QA, Evaluate, Audit, Theme, Settings, Help | `Views/MainWindow.xaml` |
| B. Toolbox | 16 tools (vector icons), brush radius | `MainWindow.xaml`, `Resources/Icons.xaml`, `Interaction/ToolController.cs` |
| C. Workspace | `ImageViewport`, view options (overlay mode, opacity, fill, boundaries, grid, minimap), flash feedback, empty state with recent projects | `Controls/ImageViewport.cs`, `Controls/Minimap.cs` |
| D. Properties | class palette with search; segmentation settings incl. hybrid stage toggles; candidate status, suggestion and intermediate stages; selected annotation (class, verification, evidence, suggestion, severity, reviewer assessment, notes, visibility, geometry, provenance, parameters, revision history with restore); region operations; annotation list; smart suggestions & batch assistance; pixel inspector | `MainWindow.xaml` |
| E. Browser & status | thumbnails, index, progress, workflow state, image metadata; status bar with dimensions, zoom, cursor, tool, unsaved indicator, autosave status, progress | `MainWindow.xaml` |

Panels are resizable (grid splitters) and collapse with a double-click on the splitter; right-panel sections are
expanders.

## MVVM

* `MainViewModel` — project lifecycle, images, palette, parameters, tool selection, view options, commands,
  autosave timer, dialogs.
* `EditorViewModel` — one per open image: wraps `AnnotationSession`, owns the `Mat` and `TiledImageSource`,
  maintains `ViewportScene`, annotation view models, selection, inspector, and background-task tracking.
* `AnnotationViewModel` — properties panel; every setter calls an undoable session method.
* Commands use CommunityToolkit.Mvvm `[RelayCommand]` (async where appropriate) with `CanExecute`.
* Code-behind is limited to view wiring (keyboard routing, panel collapse, viewport/minimap hookup).

## Rendering

`ImageViewport` (a `FrameworkElement` with `OnRender`) draws in this order: background, tiles of the image
pyramid for the current zoom (`TiledImageSource.LevelFor(scale)`, 512² tiles, LRU cache of 160 tiles),
annotation fills (per-annotation premultiplied bitmaps over their bounds; masks > 4 MP are displayed at reduced
resolution), outlines through pixel centres, candidate (cyan) and ROI, guidance strokes, tool previews, vertex
handles, pixel grid (≥ 800 %). Bitmap scaling is nearest-neighbour above 100 % (pixel-exact) and high-quality
below. Overlay modes: Normal, Hidden ("before"), Split (overlays only right of centre), Checkerboard (48-DIP
cells), ContoursOnly (high-contrast double stroke).

All pointer positions are converted with `ViewTransform.ViewToImage` before leaving the control; the tool
controller and everything below it only see image coordinates.

## Interaction

* Mouse wheel zooms at the cursor (×≈1.2 per notch); middle-drag or Space+drag pans; the Select tool pans with
  a left-drag and selects with a click.
* "1:1" sets `Scale = 1/DpiScale` so one image pixel maps to one physical pixel on the current monitor
  (per-monitor-v2 DPI awareness is declared in `app.manifest`).
* Tools are implemented as a state machine in `ToolController` (rectangle, polygon, lasso, smart contour,
  FG/BG/boundary brushes, eraser, vertex editor, split, merge, fill, colour select, magic wand, measure).
* Keyboard shortcuts come from `UserSettings.KeyMap` (`%APPDATA%/CuAnnotator/settings.json`); class shortcuts
  come from the taxonomy. When a text box has focus only Ctrl+S is global.

## Themes

`Themes/Light.xaml` (default, navy/blue accents) and `Themes/Dark.xaml` define the same brush keys; `ThemeService`
swaps merged dictionary 0 at runtime; all UI uses `DynamicResource`.
