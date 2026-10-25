# 09 — Export Formats

## Dataset directory

```
Dataset/
├── images/            source images: lossless PNG re-encode or byte-identical copies
├── masks/             semantic masks: 8-bit single-channel PNG, same W×H as the image, value = class ID
├── colorized_masks/   RGB visualisation (display colours) — never use for training
├── binary_masks/      class_<id>/ and all_corrosion/ (0/255) when requested
├── annotations/       native per-image JSON (lossless, includes taxonomy and coordinate convention)
├── coco/              annotations.json (+ annotations_train/val/test.json when splitting)
├── yolo/              images/<split>/, labels/<split>/, data.yaml
├── metadata/          classes.json, dataset.json (split, seed, checksums, calibration), verification.json
└── README.md          layout, classes, scientific note, export warnings
```

After writing, every semantic mask is **re-read and compared pixel-by-pixel** with the project (dimensions and
values); any mismatch is reported as a warning.

## Semantic masks

* 8-bit, single channel, PNG (lossless, compression level 6). No JPEG, no interpolation, no antialiasing.
* 0 = unlabeled, 255 = ignore, other values = taxonomy class IDs.
* Composition: annotations painted in ascending Z-order (ties by creation time); hidden annotations are exported
  (visibility is a display setting).

## Binary masks

0/255 PNG for: selected annotations or the active class (toolbar "Binary mask…"), any list of class IDs, or all
corrosion classes (every class except Healthy Copper (1), Unlabeled (0) and Ignore (255)).

## COCO

* `images`, `categories` (taxonomy without 0 and 255; `supercategory` from the parent category),
  `annotations` (`iscrowd = 0`).
* `area` = exact mask pixel count; `bbox` = [x, y, w, h] of the mask in pixel-edge coordinates.
* `segmentation`: polygons `[[x1, y1, …]]` in pixel-centre coordinates (x + 0.5, y + 0.5), one list per
  connected component — or **compressed RLE** `{"size": [h, w], "counts": "…"}` when the annotation has holes,
  a component smaller than 3 vertices, or when "always RLE" is selected.
* pycocotools rasterises polygons with its own rule, so polygon mode can differ from the native mask by boundary
  pixels; **RLE mode is pixel-exact**.
* Class 255 (Ignore) annotations are omitted from COCO and reported.
* A non-standard `cuannotator` object per annotation (ignored by COCO tools) carries the native ID, verification
  status and revision.

## YOLO segmentation

One line per polygon: `class x1 y1 x2 y2 …` with x_n = (x + 0.5)/W, y_n = (y + 0.5)/H (6 decimals). Class
indices are contiguous from 0 in ascending taxonomy ID order (excluding 0 and 255); `data.yaml` lists the
mapping. Limitations, each reported as a warning:

* **Holes:** cannot be represented. Policy *Skip* (default) omits the annotation; *OuterBoundaryOnly* exports the
  outer ring (hole pixels become foreground).
* **Multipart annotations:** each component becomes a separate line (instance identity is lost).
* Components with fewer than 3 vertices are skipped.

## Native JSON

Per-image documents (`CuAnnotator.ImageAnnotations`, schema v1) contain the taxonomy, coordinate convention,
rasterisation rule, image metadata and all annotations with masks, polygons, verification, evidence, revisions
and parameters. They can be read back with `NativeJsonSerializer.DeserializeImage`.

## Dataset splitting

`DatasetSplitter.Split(images, train, validation, seed, groupBySpecimen = true)`:

1. Group images by `SpecimenKey` (specimen ID, or the image itself if none).
2. Sort groups by key (ordinal), shuffle with a seeded Fisher–Yates permutation.
3. Assign each group whole to the split furthest below its target image count.

Patches of the same physical specimen therefore never appear in different splits unless grouping is disabled;
`LeakingSpecimens` reports any violation. The seed, fractions and grouping flag are written to
`metadata/dataset.json`.
