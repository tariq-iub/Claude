# 05 — Segmentation Algorithms

All algorithms implement `ISegmentationAlgorithm.SegmentAsync(SegmentationRequest, CancellationToken)`:

* **Request:** full BGR image (not copied), ROI, optional initial mask, foreground / background / barrier
  strokes (polylines with radius, image coordinates), previous GrabCut state, `SegmentationParameters`.
* **Result:** mask (image coordinates), exact polygon components, processing time, parameters actually used,
  diagnostics, named intermediate stages, warnings, GrabCut state for reuse.

Positive strokes are re-imposed as foreground and negative strokes as background **after** post-processing:
user guidance is a hard constraint. Results are always clipped to the ROI.

## A. GrabCut (`GrabCutSegmenter`, v1.0.0)

Rother, Kolmogorov & Blake (2004) via OpenCV. Colour GMMs (5 components each) for foreground and
background are fitted and a graph cut minimises

E(α, k, θ, z) = U(α, k, θ, z) + V(α, z)

(data term from GMM likelihoods; pairwise term γ·exp(−β‖z_m − z_n‖²) on neighbouring pixels).

Implementation details:

* The ROI is processed inside a **context frame** (ROI + `RoiContextMargin`·max(w,h)) so the background model
  sees real surroundings. Outside the ROI: `GC_BGD`; inside: `GC_PR_FGD`, or `GC_PR_FGD/GC_PR_BGD` from an
  initial mask (polygon/lasso ROI, re-segmentation of an existing annotation).
* Strokes: foreground → `GC_FGD`, background and barriers → `GC_BGD`.
* If the ROI covers the whole frame (no background samples), a border band of the ROI becomes `GC_PR_BGD`
  (warning issued).
* **Refinement reuses state**: when re-run on the same ROI, the previous 4-state mask and both GMMs are reused
  and only new seeds are written (`Eval` mode), so the result converges instead of restarting.
* Iterations run one at a time to allow cancellation between iterations.
* **Determinism:** `Cv2.SetTheRNG(RandomSeed)` before the k-means GMM initialisation (verified by test).
* **Quality presets:** frames above 512² (Fast) or 2048² (Balanced) pixels are processed downscaled (INTER_AREA)
  and the label map is upsampled with nearest neighbour; `Precise` always uses full resolution. A warning states
  the scale used.

Limitations: a rectangle is initialisation, not identification; similarly coloured adjacent phases need
background strokes. GrabCut favours compact regions and may close small holes (use a background stroke).

## B. CIELAB colour-guided segmentation (`LabColorSegmenter`, v1.0.0)

Each pixel p_i = [L*_i, a*_i, b*_i] (exact sRGB→CIELAB, D65; see 06). Steps:

1. Reference colour r = per-channel **median** Lab of the foreground strokes; else of the initial mask; else of
   the central 20 % of the ROI. (The source is reported in diagnostics.)
2. Candidate set C = { i : ΔE(p_i, r) ≤ T } with ΔE = CIEDE2000 (default) or CIE76.
3. Optional edge barrier: Canny edges (σ = 3×3 Gaussian, thresholds `CannyLow/High`) and barrier strokes are
   removed from C.
4. Optional spatial constraint: keep only 8-connected components of C intersecting the seed region (region
   growing); without it the result is a global colour selection (used by the colour-selection tool).
5. Post-processing.

The magic wand is this algorithm with a single-pixel seed, connectivity on, and a window of ±1024 px.

## C. Marker-controlled watershed (`WatershedSegmenter`, v1.0.0)

Meyer's flooding (OpenCV) on the Gaussian-smoothed colour image (σ = `WatershedPreBlurSigma`) with exactly two
marker labels, which prevents over-segmentation:

* Background (1): context frame outside the ROI, background strokes (+ ROI border when the ROI fills the frame).
* Foreground (2): foreground strokes; else the initial mask eroded by 2 px; else automatic markers — pixels in
  the inner half of the ROI within ΔE00 ≤ `WatershedAutoMarkerDeltaE` of the central median colour, eroded
  by 2 px, largest component.
* Barrier strokes are painted as maximal relief.

Watershed lines (−1) are assigned to background. Limitations: textured regions create many local minima; with
weak markers the basin can be small — add foreground strokes.

## D. Morphological active contours (`ActiveContourSegmenter`, v1.0.0)

Márquez-Neila, Baumela & Álvarez (2014): binary level set u evolved with morphological operators.

**Region mode (MorphACWE, Chan–Vese):** with c₁, c₀ the mean Lab colour inside/outside,
for pixels on the contour (|∇u| > 0):

u ← 1 if λ₁‖I − c₁‖² − λ₂‖I − c₀‖² < 0;  u ← 0 if > 0.

**Edge mode (MorphGAC, geodesic):** g(I) = 1/√(1 + α|∇(G_σ * I)|) on the luminance; balloon force ν dilates
(ν > 0) or erodes (ν < 0) where g > θ (θ = 40th percentile of g); attachment u ← 1 where ∇g·∇u > 0, 0 where < 0.

Both modes then apply the curvature operator alternately SI∘IS and IS∘SI `smoothing` times, where SI/IS are
sup-inf / inf-sup over the four 3-pixel line structuring elements. Iteration stops when fewer than
`ConvergenceTolerance`·|ROI| pixels change. Initialisation: initial mask, else dilated foreground strokes, else
the ROI shrunk by 10 %. Strokes and barriers are re-imposed each iteration.

| Parameter | Meaning |
|---|---|
| `ActiveContourIterations` | maximum iterations |
| `ActiveContourSmoothing` | curvature operator applications per iteration |
| `EdgeAttraction` (α), `EdgeSigma` (σ) | edge-stopping function |
| `RegionLambdaInside/Outside` (λ₁, λ₂) | region influence |
| `Balloon` (ν) | expansion/contraction in edge mode |
| `ConvergenceTolerance` | stopping criterion |

Limitations: pixel-level (no sub-pixel) contour; Chan–Vese assumes two roughly homogeneous populations.

## E. Hybrid (`HybridSegmenter`, v1.0.0)

```
ROI → [1 GrabCut] → M1 → [2 CIELAB refinement] → M2 → [3 geodesic edge refinement] → M3 → [4 post-processing] → contours
```

* Stage 1: GrabCut (or, if disabled, colour threshold around the reference colour).
* Stage 2: r = median Lab of M1; M2 = (M1 ∧ ΔE ≤ T) ∨ (dilate(M1, 2 px) ∧ ΔE ≤ T/2) ∨ foreground strokes —
  removes colour outliers accepted by GrabCut and recovers thin, very similar boundary pixels. Removed/added
  pixel counts are reported.
* Stage 3: MorphGAC (balloon 0) for `HybridEdgeIterations` from M2.
* Stage 4: post-processing and connected-component filtering.

Every stage can be toggled; all intermediate masks are returned and shown in the UI. The hybrid is **not**
assumed to be better than its constituents — use the evaluation module (Evaluate) to compare algorithms
against reference masks on your data.

## Post-processing (`MaskPostProcessor`)

Order: binarise → opening(r_o) → closing(r_c, zero-padded) → Gaussian boundary smoothing(σ) → hole filling
(only if `PreserveHoles` is off) → clip to allowed region → remove 8-connected components smaller than
`MinComponentArea` (0 disables; small microstructures are never removed unless this policy says so) →
optional keep-only-seeded-components → re-impose seeds.

## Boundary snapping (`BoundarySnapper`)

The approximate path is resampled every 2 px; each point moves along its normal within ±`SnapRadius` to the
maximum of (normalised smoothed gradient magnitude − 0.02·|offset|); offsets are median-filtered (window 5).
The snapped path is used as a barrier constraint for the candidate.

## Class-assisted suggestions (`ClassSuggester`)

Descriptor (order fixed): mean L*, a*, b*; median L*, a*, b*; std L*, a*, b*; Canny edge density; mean Sobel
gradient magnitude. Features are standardised over the project's manually labelled exemplars; distance-weighted
k-NN (k = 5) votes. **SimilarityScore = 1/(1 + d_min)** for the winning class (standardised units): an
uncalibrated similarity, explicitly not a probability. Accepting a suggestion sets `ClassAssignmentSource =
AcceptedSuggestion` and leaves verification untouched.

## Smart region suggestions (`RegionSuggester`)

On a ≤1 MP copy: for each class present (or from batch reference colours captured on another image), unlabeled
pixels within ΔE00 ≤ T of the class median colour, opened (r = 1), connected components ≥ min area, upsampled
(nearest) and restricted to unlabeled full-resolution pixels. Candidates are previewed as normal candidates and
must be accepted and labelled by the user.
