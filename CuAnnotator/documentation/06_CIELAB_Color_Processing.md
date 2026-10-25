# 06 — CIELAB Colour Processing

## Conversion used by CuAnnotator

Input: 8-bit sRGB (R, G, B) ∈ {0…255}. No ICC transform is applied (see *Calibration* below).

1. Linearise: c = v/255; c_lin = c/12.92 if c ≤ 0.04045, else ((c + 0.055)/1.055)^2.4 (exact, via a 256-entry LUT).
2. XYZ (sRGB primaries, D65):

   X = 0.412453 R + 0.357580 G + 0.180423 B
   Y = 0.212671 R + 0.715160 G + 0.072169 B
   Z = 0.019334 R + 0.119193 G + 0.950227 B

3. Normalise by the D65 white point Xn = 0.950456, Yn = 1, Zn = 1.088754.
4. f(t) = t^(1/3) if t > 0.008856, else 7.787 t + 16/116.
   L* = 116 f(Y) − 16 (Y > 0.008856) or 903.3 Y; a* = 500 (f(X) − f(Y)); b* = 200 (f(Y) − f(Z)).

The same constants as OpenCV are used, so results are comparable, but the computation is exact in double
precision (`ColorScience.RgbToLab`, `LabImage.FromBgr`).

## OpenCV representations — what to watch for

| OpenCV call | Output ranges |
|---|---|
| `cvtColor(8U BGR, BGR2Lab)` | L = L*·255/100, a = a* + 128, b = b* + 128 (all 0–255, quantised) |
| `cvtColor(32F BGR in [0,1], BGR2Lab)` | L* ∈ [0,100], a*, b* unscaled (≈ −127…127) |

`ColorScience.FromOpenCv8Bit` decodes the 8-bit form. OpenCV's float path approximates the sRGB gamma with a
spline; the test `OpenCvFloatLab_DeviatesSlightlyFromExactFormula_DocumentedBound` measures the deviation
(< 1 ΔE76, typically ≈ 0.2). CuAnnotator therefore does not use OpenCV for quantitative Lab values.

## Colour differences

* **CIE76:** ΔE*ab = ‖p − q‖₂.
* **CIEDE2000** (default): Sharma, Wu & Dalal (2005) with k_L = k_C = k_H = 1, including the G-factor a′
  adjustment, mean-hue discontinuity handling and the rotation term R_T:

  ΔE00 = √[(ΔL′/S_L)² + (ΔC′/S_C)² + (ΔH′/S_H)² + R_T (ΔC′/S_C)(ΔH′/S_H)]

  Verified against **all 34 reference pairs** of Sharma et al. to 4 decimals, including the pairs that test the
  hue-angle discontinuity (pairs 7–16), plus symmetry.

## Pixel inspector

Shows coordinate (and µm if calibrated), RGB, HSV (H in degrees, S and V in [0,1]), CIELAB (D65, sRGB assumed),
grey level (Rec. 601 weights) and the current label.

## Representative colours

Region colours are summarised by the per-channel **median** in Lab (robust to boundary contamination and
specular highlights); `RegionStatistics` also reports means, standard deviations, min/max, and histograms
(L* 20 bins over [0,100]; a*, b* 32 bins over [−128,128]).

## Calibration and scientific caveats

* Without colour calibration, camera RGB is not a measurement of surface reflectance; illumination, white
  balance, exposure, specular reflection and compression all shift Lab values.
* Embedded ICC profiles are **detected and recorded** (`ImageItem.ColorProfile`) but not applied in this
  version. Images with non-sRGB profiles should be converted to sRGB before import if colour statistics are to
  be compared across acquisitions.
* 16-bit images are linearly scaled to 8 bits (v/257) for segmentation and display; the original file is kept.
* Colour similarity is used to **propose regions** and **suggest classes**, never to establish chemical identity.
  Verification status and evidence are recorded separately.
