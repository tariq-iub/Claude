# Evaluation protocol (declare this before opening the test set)

## Reference
Only **adjudicated expert masks** (`reference_type=adjudicated_expert`) support accuracy claims. Any other reference
(single expert, pseudo-labels, synthetic data) marks the report **PROVISIONAL** automatically.

## Ignored pixels
Reference pixels with value 255 are excluded. A prediction of 255 on a valid reference pixel (an abstention) is an
error in every accuracy-type metric. Abstentions are also reported separately (`abstention_rate`) and shown in coverage–risk curves.

## Metrics (`evaluation/metrics.py`)
- **Primary:** macro mean IoU over the 6 classes, computed per image and then averaged across images.
- **Secondary:** binary corrosion Dice, boundary F1 (2 px tolerance), rare-class IoU, and ECE.
- **Also reported:** per-class IoU, Dice, precision and recall; frequency-weighted Dice; balanced accuracy; the confusion matrix; Healthy IoU
  and corrosion-class mIoU kept separate; sensitivity, specificity, FPR and FNR; HD95; recall of small regions (≤ 200 px);
  Brier score; coverage–risk curves; time per image and fit time; peak Python memory; and mIoU under 7 acquisition perturbations.
- **Rare classes:** corrosion classes below 5% of the valid **training** pixels, fixed before testing.

## Statistics (`evaluation/statistics.py`)
- Paired comparisons on the same test images. The primary metric is resampled at image level with 2,000 bootstrap iterations,
  giving a mean difference, a 95% CI and a two-sided p-value. The Wilcoxon signed-rank test serves as a check.
  Holm correction is applied across all pairs, and effect size is reported as d_z.
- No p-value is reported with fewer than 10 paired images. With fewer than 30, the report adds a power warning.
- **Clustering:** if several test images come from the same specimen, resample by specimen. To do this, aggregate the per-image
  values to specimen means before calling `compare_approaches`.

## Selection rule (predeclared)
Weighted rank aggregation (`select_best`): mIoU 0.30, binary Dice 0.15, boundary F1 0.15, rare-class IoU 0.15,
ECE 0.05 (lower is better), review fraction 0.10 (lower is better), seconds per image 0.05 (lower is better), robustness 0.05.
A method is called "best" only if it ranks first **and** its mIoU CI against the runner-up excludes 0. Otherwise the report
says that the methods cannot be told apart, and gives the winner under each constraint (accuracy, cost, effort).

## Active-learning comparison
Compare random, uncertainty and disagreement selection with **the same per-round budget and the same initial set**.
Report validation mIoU against the number of reviewed images, plus expert minutes when they are logged. Stop at the budget, or after
`patience` rounds that each gain less than `min_improvement`.
