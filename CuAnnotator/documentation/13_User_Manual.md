# 13 — User Manual

## 1. Getting started

1. Start **CuAnnotator.exe**. The first launch shows a short onboarding guide (Help → onboarding at any time).
2. **New project** (toolbar): name, folder, researcher, default algorithm, overlap policy. Imported images are
   copied into the project folder by default; originals are never modified.
3. **Import** images (JPEG, PNG, TIFF incl. 16-bit, BMP). Optionally enter a specimen ID, acquisition source,
   magnification and pixel size (µm/px). Images of the same specimen stay in the same dataset split.
4. Select an image in the bottom browser.

## 2. Navigating

| Action | How |
|---|---|
| Zoom | mouse wheel (zooms at the cursor), Zoom +/−, Ctrl +/− |
| Pan | Space + drag, middle-button drag, or the Select tool |
| Fit / 100 % | F / Ctrl+1 (100 % = one image pixel per screen pixel) |
| Pixel grid | shown automatically from 800 %, toggle in the view bar |
| Minimap | bottom-right navigator; click/drag to move |
| Compare | view bar → Normal / Hidden (before) / Split / Checkerboard / ContoursOnly; opacity slider |

The status bar shows image size, zoom, cursor pixel coordinates, active tool and save state. The **pixel
inspector** shows RGB, HSV, CIELAB and the current label under the cursor.

## 3. Annotating a region (core workflow)

1. Choose an algorithm in the toolbar (GrabCut is a good default).
2. Press **R** and drag a *loose* rectangle around the corrosion region. The candidate appears in cyan.
3. Refine:
   * **B** (foreground brush): paint pixels that are missing.
   * **N** (background brush): paint pixels that should be excluded (e.g. an adjacent similar phase or an exposed
     copper island → creates a hole).
   * **K** (boundary brush): trace roughly along an edge; the path snaps to the edge and blocks leakage.
   * With *Live preview* on, the segmentation re-runs after each stroke; otherwise press **G**.
4. Adjust parameters in *Segmentation settings* if needed (threshold, smoothing, minimum region area — set 0 to
   keep microscopic structures, preserve holes, quality preset).
5. Press **Enter** to accept with the active class, or a class key (**1–8**, **I** for Ignore) to accept with
   that class. Overlaps are resolved by the project's overlap policy (or a prompt).
6. In *Selected annotation* set the **verification status**. Use **Add verification evidence** to record XRD,
   Raman, FTIR, SEM-EDS, XPS, laboratory notes or literature. "Instrumentally verified" requires instrumental
   evidence.

Other ways to start a candidate: **P** polygon ROI (click vertices, double-click/Enter), **L** freehand lasso,
**S** smart contour (hybrid algorithm), **W** magic wand (click), **C** colour selection (click a sample).

## 4. Editing accepted annotations

Select with a click (Ctrl+click for several). Then:

| Tool / button | Effect |
|---|---|
| B / N with a selection and no candidate | add / remove pixels |
| E eraser | remove pixels |
| A vertex editor | drag vertices, double-click an edge to add, right-click to delete, Ctrl+drag to move |
| X split | draw a cut line; each part becomes an annotation |
| M merge | Ctrl+click several, Enter |
| H fill | fill interior holes |
| Region operations | delete, merge, union, intersect, subtract, fill holes, duplicate, re-segment |
| 1–8 / I | relabel the selection |
| Revision history | restore any earlier revision (as a new revision) |

Everything is undoable (**Ctrl+Z / Ctrl+Y**).

## 5. Classes

The palette lists classes (search by name, formula or category). **Classes** (toolbar) opens the taxonomy
editor: add classes, rename, recolour, set shortcuts, parent category, required verification, enable/disable.
IDs in use cannot be removed or renumbered; 0 (Unlabeled) and 255 (Ignore) are reserved.

## 6. Assistance (optional)

* **Class suggestions:** after segmentation the panel may show "Suggested class … similarity 0.83
  (uncalibrated)". It is a visual comparison with your earlier annotations — not a probability and not a
  verification. Accepting it records that the class came from a suggestion.
* **Find region suggestions:** proposes unlabeled regions similar in colour to classes already annotated.
* **Batch reference:** on an annotated image click "Use this image as batch reference"; on other images "Find
  region suggestions" uses those reference colours. Every proposal still needs acceptance on its own image.

## 7. Review and quality control

* **Review** (toolbar): queue filterable by class, verification, image state, text (image/reviewer/author) and
  date; double-click to jump to the annotation.
* Image workflow state (browser panel): Unannotated → In progress → Annotated → Pending review → Verified.
* **QA**: coverage (labelled / ignored / unlabeled), class distribution (pixels, instances, mean/median area),
  size histogram, verification counts, validation issues (undefined/disabled classes, empty masks, out-of-bounds
  geometry, self-intersecting edited polygons, overlaps, missing image files, verification below the class
  requirement), inter-annotator agreement when several annotation sets exist, and annotation efficiency. CSV
  export available.
* **Evaluate**: with an ROI drawn, compare all algorithms against a reference mask PNG (IoU, Dice, precision,
  recall, boundary F1 at 2 px, time).
* **Audit**: chronological log distinguishing `user:` actions from `algorithm:` proposals.

## 8. Saving and exporting

* **Ctrl+S** saves atomically (a backup of the previous version is kept). Autosave writes recovery data; after
  a crash you are offered to restore it.
* **Export** writes a dataset (semantic masks, optional colourised and binary masks, native JSON, COCO, YOLO,
  metadata, README) with a reproducible, specimen-grouped train/validation/test split. The export report lists
  every warning (e.g. YOLO cannot represent holes).
* Region operations → **Export mask…** writes the current image's class-ID PNG; **Binary mask…** a 0/255 mask;
  **Region statistics (CSV)** colour and shape statistics per annotation.

## 9. Keyboard shortcuts (defaults)

| Key | Action | Key | Action |
|---|---|---|---|
| Ctrl+O | open project | Ctrl+S | save |
| Ctrl+Z / Ctrl+Y | undo / redo | Enter / Esc | accept / cancel |
| V R P L S | select, rectangle, polygon, lasso, smart contour | B N K E | FG, BG, boundary brush, eraser |
| A X M H | vertex editor, split, merge, fill | C W U | colour select, wand, measure |
| G | run / refine | F, Ctrl+1 | fit, 100 % |
| O | toggle overlays | Delete | delete selection |
| 1–8, I | assign class / Ignore | Tab / Shift+Tab | next / previous image |
| Space+drag | pan | F1 | help |

Shortcuts are configurable in `%APPDATA%/CuAnnotator/settings.json` (`KeyMap`).
