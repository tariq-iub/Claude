# 04 — Annotation Data Model and Coordinate Conventions

## Canonical coordinate system

* Origin at the **top-left** corner of the (orientation-corrected) image.
* **X** increases to the right, **Y** increases downward, integer pixel indices are **zero-based**.
* Pixel `(x, y)` covers the continuous square `[x, x+1) × [y, y+1)`; its centre is `(x + 0.5, y + 0.5)`.
* A continuous image position `(u, v)` lies in pixel `(⌊u⌋, ⌊v⌋)` (floor rule, also for negative values).

### Transform chain (display → image)

| Space | Unit | Conversion |
|---|---|---|
| Physical screen | device pixels | WPF divides by the monitor DPI scale `s_dpi` (1.0 at 96 DPI, 1.5 at 144 DPI) |
| Viewport | DIPs | `view = image × Scale + Offset` (`ViewTransform`) |
| Image (continuous) | image pixels | `image = (view − Offset) / Scale` |
| Image pixel | index | `floor(image)` |
| Raster pixel centre | — | `index + 0.5` (used to draw polygons and in COCO/YOLO exports) |
| Polygon vertex | index | integer index of a boundary pixel |

Zoom, pan, window size and DPI only change `Scale`/`Offset` or the WPF DIP scale; stored geometry is always
in image indices. `ViewTransformTests` verify that a point drawn at 25 % zoom maps to the same image pixel after
arbitrary pan/zoom, that zoom-at-cursor keeps the anchor fixed, and that "100 %" equals one image pixel per
*physical* pixel (`Scale = 1/s_dpi`).

## Entities

```
AnnotationProject (schemaVersion = 1)
 ├── Id, Name, Description, Researcher, Institution, CreatedAt, ModifiedAt, app versions
 ├── Taxonomy (Name, Version, Classes[])
 │    └── CorrosionClass: Id (0–255, permanent), Name, ChemicalFormula, Description, ColorHex, Shortcut,
 │                        IsEnabled, ParentCategory ("Copper Surface/Corroded/Oxides"), RequiredVerification, IsReserved
 ├── ProjectSettings: DefaultAlgorithm, DefaultParameters, OverlapPolicy, AutosaveIntervalSeconds,
 │                    CopyImagesIntoProject, SeverityLevels, SeverityCriteria, ExportPreferences
 └── Images[]: ImageItem
      ├── Id, FileName, RelativePath, OriginalSourcePath, Width, Height, PixelFormat, BitDepth, Channels,
      │   FileSizeBytes, Sha256, ImportedAt, ColorProfile, ExifOrientation, ExifOrientationApplied,
      │   SpecimenId, AcquisitionSource, Magnification, MicrometersPerPixel, Notes, WorkflowState
      └── Annotations[]: Annotation
           ├── Id (GUID, stable), ImageId, ClassId, DisplayName, ClassAssignmentSource
           ├── Mask : RasterMask  ← authoritative geometry
           ├── Polygons[] : PolygonComponent { Outer, Holes[] }  ← editable vector view
           ├── GeometrySource (Segmentation, PolygonEdit, BrushEdit, Merge, Split, BooleanOperation, Import)
           ├── BoundingBox, PixelArea, CentroidX/Y, Perimeter  (derived)
           ├── SegmentationAlgorithm, AlgorithmVersion, AlgorithmParameters{}  (+ efficiency.* entries, roi)
           ├── VerificationStatus, Evidence[] {Type, Source, Notes, Date}, ReviewedBy, ReviewedAt
           ├── Notes, Severity, ReviewerConfidence (human assessment 0–1)
           ├── Suggestion {ClassId, SimilarityScore (uncalibrated), Basis}, Descriptor[]
           ├── CreatedAt, ModifiedAt, CreatedBy, Revision, RevisionHistory[], IsVisible, ZOrder, AnnotationSet
```

* IDs are GUIDs; array positions are never used as identity.
* `AnnotationSet` separates independent annotators ("default", "annotator-B" …) for agreement analysis.
* `VerificationStatus` ∈ {Unverified, VisuallyEstimated, ExpertReviewed, InstrumentallyVerified}. It is changed
  only by explicit user action; class suggestions and algorithms never alter it.

## RasterMask

`RasterMask(Bounds, Data)`: a 0/1 byte array covering only `Bounds` (row-major). Operations (`Union`,
`Intersect`, `Subtract`, `Reframe`, `Trim`, `ClipToImage`, `OverlapCount`, `Centroid`) return new instances.
Equality is pixel-set equality, independent of padding.

## Mask → polygon

OpenCV border following with `RETR_CCOMP` and `CHAIN_APPROX_NONE` on a 1-pixel padded copy:
top-level contours are outer boundaries of 8-connected components; their children are holes; islands inside
holes become separate components. Every boundary pixel is a vertex, so the conversion is lossless.

## Rasterisation rule (polygon → mask)

1. Vertices are integer pixel indices (pixel centres).
2. Components are processed largest outer ring first (islands inside holes are drawn after the hole).
3. The outer ring is scan-filled (8-connected edges) and its boundary pixels are set.
4. Each hole ring is filled with 0, then its boundary pixels are set to 1 again (hole rings produced by border
   following pass through foreground pixels).

For unsimplified rings this reproduces the source mask exactly (10 randomised round-trip tests, plus tests with
holes, islands and image-border contact). After a *simplified* polygon is edited, the new mask is the
rasterisation of the edited polygon; the exact previous mask remains in `RevisionHistory`.

## Overlap policy

| Policy | Effect on accept |
|---|---|
| PreserveExisting (default) | new mask ← new − ∪existing |
| ReplaceExisting | each overlapping existing mask ← existing − new (revision recorded; emptied annotations removed) |
| AllowOverlapTopmostWins | both kept; semantic export paints in ascending `ZOrder` (ties: creation time); QA reports overlap pixels |
| AskUser | the UI asks and applies one of the above |

## Revisions

Every geometry, class or restore operation on an accepted annotation pushes a snapshot of its previous state
(`Revision`, timestamp, author, reason, class, verification, geometry source, mask, notes) and increments
`Revision`. `RestoreRevision` re-applies an older state as a new revision (history is never rewritten).
