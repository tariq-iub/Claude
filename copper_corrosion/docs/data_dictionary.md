# Data dictionary

The machine-generated audit is in `docs/audit/audit.md` and `audit.json`. To regenerate it, run `cucorr audit`.

## settings.xlsx, sheet `settings` (6 rows)
| Column | Type | Meaning (as established by the audit) |
|---|---|---|
| `id` | int 1–6 | Class key. **Project class ID = id − 1** (assumption A2, checked during parsing) |
| `title` | str | Class name: Healthy, Cu2O, CuO, CuCl, CuCl2, Patina. These match `labels.xlsx` exactly |
| `rep_color` | `"R,G,B"` str | Representative sRGB colour |
| `ref1..ref3` | `"R,G,B"` str | Three more example colours **of the same class**. The parser treats all four as equal prototypes (assumption A3) |

How values are parsed: 22 of the 24 cells end in `\n`, and the parser strips it. A malformed or out-of-range value raises an error; the parser does not guess. No values are missing. A missing cell would become `None` and would not be counted as a prototype.

**Check this entry:** CuCl2 `ref1 = (31,38,42)` is near-black. It is ΔE00 = 75 from the CuCl2 `rep_color` (143,235,235), and it lies close to the CuO references. Either it shows dark CuCl2 deposits or it was entered by mistake. Until an expert decides, it stays in as recorded. To test its effect, run the ablation with `use_manifest_prototypes: false` and edit the workbook.

## labels.xlsx, sheet `manifest` (660 rows)
| Column | Type | Meaning |
|---|---|---|
| `version` | int (always 3) | Annotation-set version. All rows share it |
| `cluster_id` | int 1–697 | Colour-cluster identifier. **Not an image ID.** There are 37 gaps and no duplicates |
| `lab_l, lab_a, lab_b` | float | CIELAB centroid of the cluster. L* runs 2.9–99.0, a* −34.0–26.0, b* −19.9–50.0. Every row lies inside the sRGB gamut |
| `annotation_title` | str | Class the expert assigned to the cluster |
| `NAVICAT_ROWID` | int | Database export key. Ignored |

Class counts: Healthy 267, CuCl2 86, Patina 82, CuCl 79, CuO 74, Cu2O 72.

### How the manifest should be read
Nothing in the manifest points to an image or a pixel. The most defensible reading is
**palette-level weak supervision**: someone clustered the pixel colours of the image collection, and an
expert labelled each cluster centroid. The project therefore uses the manifest rows **only as more
colour prototypes** (Approach A) and never as pixel ground truth. Any mask built from them is a *pseudo-label*.

### Colour-convention check (assumption A1)
The manifest Lab values were classified by their nearest settings reference colour, with both converted
using sRGB → CIELAB (D65, 2°) and compared with CIEDE2000. This reproduces the expert label for 86.2% of clusters. If the white points
differed (for example D50 against D65), we would expect a systematic hue shift and much lower agreement. The
check therefore *supports* A1 but does not prove it. The disagreements are concentrated in **CuCl**. Manifest
CuCl clusters are greenish (mean a* = −7.9), and 30 of the 79 lie nearest to a Patina reference. The settings CuCl references are near-white.
This is a real overlap in colour space, and colour alone cannot resolve it.

## Mask conventions (output)
| File | Encoding |
|---|---|
| `multiclass_mask.png` | uint8, 0–5 = class ID, **255 = Unknown/Uncertain (ignored)** |
| `binary_mask.png` | uint8, 0 = Healthy, 1 = any corrosion class, **255 = Unknown/Uncertain** (never counted as healthy) |
| `uncertainty_map.png` | uint8, normalised entropy × 255 |
| `probabilities.npy` | float16 (H, W, 6) |
| Reference masks (`data/annotations/...`) | same as `multiclass_mask.png` |

## Information still needed
1. The images (~500) and, for each one, a **specimen/sample/session ID** for grouped splitting.
2. Dense expert masks for a reference subset. None have been supplied, so no accuracy can be reported yet.
3. Where the manifest came from: the clustering method, the colour space and white point, and the images it was computed on.
4. Camera model and settings, lighting, any colour chart, and the px/mm calibration.
5. Whether CuCl2 `ref1` is correct.
6. Any XRD, Raman, XPS or EDS characterisation of specimens, which would be needed for chemical validity.
