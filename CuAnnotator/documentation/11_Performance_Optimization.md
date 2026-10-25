# 11 — Performance and Optimisation

## Measured results

Numbers below are **measurements**, produced by `benchmarks/CuAnnotator.Benchmarks` (median of 3 runs after a
warm-up; synthetic images of ellipses + Gaussian noise). The full table with machine details is committed in
`benchmarks/results/`. They were measured on the Linux development VM (4 logical CPUs, 15 GB RAM, no GPU),
**not** on the target Intel Core i7 10th-gen Windows workstation — re-run `scripts/benchmark.ps1` there.

| Operation | 4K (3840×2160) | 8K (7680×4320) |
|---|---:|---:|
| Decode PNG + SHA-256 + metadata | ~0.26 s | ~0.96 s |
| Exact sRGB→CIELAB, full image | ~0.27 s | ~0.61 s |
| CIEDE2000 map, 1024² | ~0.06–0.11 s | |
| GrabCut, ROI 512², Precise (full res.) | ~4.0 s | ~3.3 s |
| GrabCut, ROI 1024², Balanced | ~4.5 s | ~4.4 s |
| GrabCut, ROI 2048², Balanced / Fast | ~2.2 s / ~0.9 s | ~6.9 s / ~5.6 s |
| GrabCut, ROI 1024², Precise | **~65 s** | |
| CIELAB segmentation, ROI 800² | ~0.10 s | |
| Watershed, ROI 800² | ~0.10 s | |
| Active contour (150 it. max), ROI 800² | ~1.4 s | |
| Hybrid, ROI 800² | ~4.9 s | |
| Compose label map, 200 annotations | 0.04 s | 0.08 s |
| Write class-ID PNG | 0.13 s | 0.31 s |
| Project save (JSON + SQLite), 200 annotations | 0.63 s | 0.45 s |
| Project load, 200 annotations (11.6 MB JSON) | 0.24 s | 0.24 s |

| COCO export, 200 annotations (auto / always-RLE) | 0.11 s / 0.08 s | 0.08 s / 0.07 s |


## Findings and actions taken

* **GrabCut dominates interactive latency** and its cost grows faster than linearly with the processed frame
  (512² ROI ≈ 4 s, 1024² ROI ≈ 65 s at full resolution on this VM). OpenCV's GrabCut is single-threaded.
  Action: the `Balanced` preset (default) now caps the processed context frame at 768² pixels (`Fast`: 512²),
  downscaling larger frames (INTER_AREA) and upsampling the label map with nearest neighbour; a warning reports
  the scale. `Precise` keeps full resolution for final boundaries on small ROIs. Timing still depends strongly on
  image content (number of graph-cut augmenting paths), which explains the 4K/8K differences at equal frame size.
* **COCO RLE encoding** originally visited every image pixel for every annotation (9.1 s for 200 annotations on
  8K). The encoder now adds the zero runs outside each mask's bounds arithmetically and visits only the bounded
  pixels (identical output, tested against the generic encoder): 9.1 s → 0.07 s on 8K.
* The **hybrid** pipeline costs GrabCut + an edge-refinement contour run; use `Fast` quality for exploration.

## Memory

* The full-resolution image is decoded once (8-bit BGR: 25 MB for 4K, 100 MB for 8K) and shared by display,
  segmentation and statistics; it is disposed only when no background task still reads it.
* Display uses a lazily built pyramid (each level ¼ of the previous: +33 % at most) and a 160-tile LRU cache
  of 512² frozen bitmaps (≤ ≈ 120 MB), so no full-size WPF bitmap is ever created.
* Annotation masks are stored only over their bounds; overlay bitmaps for masks > 4 MP are rendered at reduced
  resolution (display only).
* Segmentation works on the ROI/context frame, never the whole image (except explicit colour selection or
  region suggestions, which use the visible region or a ≤ 1 MP copy).
* The benchmark process (4K and 8K images, 200-annotation projects, full exports) peaked at the working set
  reported in the results file.

## Responsiveness

* Segmentation, decoding, export and thumbnails run on the thread pool; the UI thread only applies results.
* GrabCut runs one iteration per call so cancellation (Esc, new stroke) is honoured between iterations; active
  contours check cancellation every iteration. New input during a run supersedes it (stale results discarded).
* A progress bar is shown during segmentation, import and export.

## Tuning guidance

| Situation | Recommendation |
|---|---|
| Large ROI, slow feedback | `Fast` while refining, `Precise` for the final run on small ROIs |
| Many small particles | CIELAB colour selection / magic wand (~0.1 s) instead of GrabCut |
| Very large images (> 8K) | navigation remains tile-based; keep ROIs local |
| Batch region suggestions | operate on a ≤ 1 MP copy by design |

## Not measured here

WPF rendering performance (tile drawing, overlay compositing) requires Windows and was not measured. The
design bounds per-frame work to the visible tiles and per-annotation bitmaps.
