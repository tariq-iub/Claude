# 14 — Research Methodology

## Purpose

CuAnnotator supports the creation of pixel-level ground truth for multiclass copper-corrosion segmentation with
a human-in-the-loop workflow:

**Image → ROI → automatic contour → interactive refinement → phase assignment → verification → persistence → export**

Its contribution is a *practical, reproducible workflow* that reduces manual contour tracing while keeping the
annotator in control — not a claim that chemical phases can be identified reliably from colour.

## Separation of evidence levels

| Level | Recorded as | Produced by |
|---|---|---|
| Algorithmic boundary proposal | audit entries with actor `algorithm:<name>@<version>`, candidate stages | segmentation algorithms |
| Visual class prediction | `Annotation.Suggestion` (uncalibrated similarity, basis text) | class-assisted subsystem |
| Annotator label | `ClassId`, `ClassAssignmentSource` | the user |
| Expert review | `VerificationStatus = ExpertReviewed`, `ReviewedBy/At`, evidence | a reviewer |
| Instrumental verification | `InstrumentallyVerified` + evidence (XRD, Raman, FTIR, SEM-EDS, XPS, publication) | a reviewer with analytical data |

Exports carry this information (`metadata/verification.json`, COCO extension field) so that downstream
studies can, for example, train only on expert-reviewed regions or report results stratified by verification.

Taxonomy notes: brochantite is a basic copper **sulfate**; bronze disease is associated with copper chlorides
and basic copper chlorides (atacamite group) and is not equated with CuCl₂. Classes have a "required
verification" level and QA flags annotations below it.

## Evaluating segmentation quality

Where reference masks exist (e.g. carefully hand-traced regions or another annotator's labels):

* IoU = |A ∩ B| / |A ∪ B|, Dice = 2|A ∩ B| / (|A| + |B|), precision = TP/(TP+FP), recall = TP/(TP+FN).
* **Boundary F1** (BF score, Csurka et al. 2013): boundary pixels of each mask (foreground pixels with a
  4-neighbour in background or outside the frame) are matched within a Euclidean tolerance τ (default 2 px,
  computed with an exact distance transform); BF = harmonic mean of boundary precision and recall. τ must be
  reported with results.
* Multiclass agreement: pixel agreement, Cohen's κ = (p_o − p_e)/(1 − p_e), per-class IoU/Dice; pixels with 255
  in either map and pixels unlabeled in both are excluded.

The *Evaluate* command runs every algorithm on the same ROI against a reference and exports CSV. Recommended
protocol: choose a stratified sample of regions per class and image condition (illumination, magnification),
fix parameters before evaluation, report median and inter-quartile range, and do not tune on the test regions.

## Measuring annotation efficiency

Recorded automatically for every accepted annotation (`AlgorithmParameters`):

| Key | Meaning |
|---|---|
| `efficiency.interactions` | ROI + guidance strokes |
| `efficiency.refinementRuns` | segmentation runs |
| `efficiency.seconds` | time from ROI to acceptance |
| `efficiency.firstProposalRetained` | |first automatic proposal ∩ accepted| / |first proposal| |

Plus counts of accepted candidates vs. edited/manual annotations (QA report) and the full audit trail. A
defensible efficiency study compares the same images annotated (a) manually (polygon/brush tools, Manual
geometry) and (b) semi-automatically, by the same annotators in counter-balanced order, measuring time and
interactions per region and the agreement (IoU, BF) of both against an adjudicated reference. **No efficiency
improvement is claimed by this software; it provides the instrumentation to measure it.**

## Dataset preparation

* Specimen-grouped splits prevent leakage of patches of one physical specimen across train/test.
* Splits are seeded and recorded; checksums of all source images are written to `metadata/dataset.json`.
* `Ignore` (255) marks pixels to exclude from training/evaluation (e.g. scale bars, out-of-focus areas).

## Known limitations

* Colour depends on acquisition; without calibration, CIELAB statistics are only comparable within consistent
  acquisition settings. ICC profiles are recorded but not applied.
* GrabCut and active contours operate at pixel resolution; sub-pixel boundaries are not represented.
* Class suggestions use simple, transparent features (colour, texture proxies); they are not validated
  classifiers.
