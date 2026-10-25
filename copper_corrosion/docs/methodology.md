# Methodology

## Shared contract
Every approach subclasses `cucorr.segmentation.base.Segmenter`:

```
fit(images, masks) -> self                 # masks: uint8 class IDs, 255 ignored
predict_proba(image) -> (H, W, 6) float32  # sums to 1
predict_mask(image) -> (mask, uncertainty, proba)   # mask uses 255 for abstention
```
Every approach goes through the same output writer (`save_annotation_outputs`) and the same benchmark
(`run_benchmark`).

## Approaches
| | Module | Supervision | Uncertainty | Hardware |
|---|---|---|---|---|
| A: reference colour matching | `color_match.py` | none (prototypes from both workbooks). Optional threshold tuning on **train** masks | distance margin, softmax score (not calibrated) | CPU |
| B: pixel ML (RF/SVM/HGB) | `pixel_ml.py` | pixel samples from train masks, class-stratified | class probabilities | CPU |
| C: SLIC + region RF + edge-aware graph smoothing | `superpixel.py` | region labels (purity ≥ 0.6) | smoothed probabilities | CPU |
| D: U-Net / DeepLabV3+ | `deep.py` | reference masks, or pseudo-labels (flagged `provisional`) | MC dropout | GPU recommended. Falls back to a tiny U-Net without SMP |
| E: SAM 2 regions + class assigner | `foundation.py` | inherits the assigner's | region-averaged | GPU, plus a checkpoint and the `sam2` package |
| F: hybrid ensemble + active learning | `hybrid.py`, `active_learning/loop.py` | starts from A, then uses reviewed masks | entropy, component disagreement | CPU (more with D/E) |

### Approach A details
- Prototypes: 4 settings colours per class, plus 660 manifest centroids (configurable).
- Distances: RGB Euclidean (naive baseline), CIE76, or CIEDE2000 (scikit-image implementation, checked against
  Sharma et al. 2005 test pair 1 in the tests).
- Aggregation: `nearest` takes the minimum over a class's prototypes; `mean` uses the class mean prototype.
- Abstention: the pixel is marked uncertain if the best distance is above `max_distance`, or if the gap between the best and second-best class is below
  `abstain_margin`. `fit()` grid-searches both thresholds on **training** masks only.
- Optional preprocessing: gray-world or max-RGB white balance, and L* illumination flattening. These are ablations.
- Performance: distances are cached in a 7-bit-per-channel RGB lookup table (max quantisation error is 1 RGB unit). The tests
  check that the result agrees with the exact computation.

### Why SAM-type models cannot separate the classes on their own
Promptable models produce class-agnostic regions that follow visible edges. Corrosion products often
grade into one another without a sharp edge. Examples are the Cu2O → CuO darkening and the green-family CuCl / CuCl2 / Patina,
which also overlap in colour (see the audit). So the boundaries from E can be good while the class labels are wrong,
and the two must be evaluated separately: boundary F1 on the region partition, and per-class IoU on the labels.

## Leakage prevention
- Splits are made by **specimen group** (`data/splits.py`), and `check_no_leakage` enforces this.
- Rare classes, thresholds, early stopping and hyper-parameters use train/val data only.
- The test masks are read only inside `run_benchmark`, after `fit`.

## Scientific caution
Class labels describe **expert-informed visual categories**. Colour is consistent with particular
corrosion products, but it does not identify them chemically. For example, nantokite (CuCl), the basic copper chlorides and the
basic sulfates/carbonates of patina can all look green. A chemical claim needs XRD, Raman or XPS on a characterised subset.
