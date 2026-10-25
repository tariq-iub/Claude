# Benchmark report

> **PROVISIONAL**: reference masks are `synthetic`, not adjudicated expert annotations. These numbers do not measure real corrosion-segmentation accuracy.

Train images: 24 | Test images: 8 | Rare classes: CuO

| Approach | mIoU [95% CI] | Corrosion mIoU | Healthy IoU | Binary Dice | Boundary F1 | Rare IoU | ECE | Abstain | Robust mIoU | s/img |
|---|---|---|---|---|---|---|---|---|---|---|
| A_rgb_euclid | 0.626 [0.584, 0.667] | 0.664 | 0.943 | 0.855 | 0.788 | 0.789 | n/a | 0.041 | 0.572 | 0.282 |
| A_lab_de2000 | 0.596 [0.575, 0.619] | 0.586 | 0.941 | 0.776 | 0.728 | 0.709 | n/a | 0.102 | 0.557 | 1.217 |
| B_rf | 0.872 [0.797, 0.932] | 0.971 | 0.993 | 0.992 | 0.939 | 0.924 | 0.003 | 0.000 | 0.712 | 0.188 |
| B_hgb | 0.876 [0.800, 0.937] | 0.975 | 0.995 | 0.995 | 0.952 | 0.949 | 0.003 | 0.000 | 0.714 | 0.096 |
| C_slic_400 | 0.712 [0.680, 0.743] | 0.659 | 0.918 | 0.896 | 0.827 | 0.757 | 0.055 | 0.004 | 0.626 | 0.242 |
| F_hybrid | 0.918 [0.857, 0.964] | 0.981 | 0.995 | 0.993 | 0.983 | 0.961 | 0.119 | 0.001 | 0.790 | 1.910 |

## Per-class IoU (pooled over test pixels)

| Approach | Healthy | Cu2O | CuO | CuCl | CuCl2 | Patina |
|---|---|---|---|---|---|---|
| A_rgb_euclid | 0.943 | 0.538 | 0.745 | 0.631 | 0.752 | 0.671 |
| A_lab_de2000 | 0.941 | 0.584 | 0.685 | 0.511 | 0.579 | 0.631 |
| B_rf | 0.993 | 0.965 | 0.898 | 0.962 | 1.000 | 1.000 |
| B_hgb | 0.995 | 0.967 | 0.929 | 0.965 | 1.000 | 1.000 |
| C_slic_400 | 0.918 | 0.656 | 0.698 | 0.734 | 0.682 | 0.560 |
| F_hybrid | 0.995 | 0.968 | 0.953 | 0.981 | 0.999 | 0.999 |

## Paired comparisons (primary metric, image-level bootstrap, Holm-corrected)

| Pair | Mean diff [95% CI] | p (Holm) | Wilcoxon p | d_z | n |
|---|---|---|---|---|---|
| A_lab_de2000 vs A_rgb_euclid | -0.030 [-0.067, 0.014] | n/a | 0.383 | -0.47 | 8 |
| A_lab_de2000 vs B_hgb | -0.280 [-0.348, -0.205] | n/a | 0.008 | -2.46 | 8 |
| A_lab_de2000 vs B_rf | -0.276 [-0.343, -0.203] | n/a | 0.008 | -2.47 | 8 |
| A_lab_de2000 vs C_slic_400 | -0.116 [-0.149, -0.075] | n/a | 0.008 | -2.12 | 8 |
| A_lab_de2000 vs F_hybrid | -0.322 [-0.372, -0.263] | n/a | 0.008 | -3.62 | 8 |
| A_rgb_euclid vs B_hgb | -0.250 [-0.309, -0.183] | n/a | 0.008 | -2.43 | 8 |
| A_rgb_euclid vs B_rf | -0.246 [-0.303, -0.181] | n/a | 0.008 | -2.45 | 8 |
| A_rgb_euclid vs C_slic_400 | -0.086 [-0.137, -0.037] | n/a | 0.016 | -1.09 | 8 |
| A_rgb_euclid vs F_hybrid | -0.292 [-0.348, -0.229] | n/a | 0.008 | -2.91 | 8 |
| B_hgb vs B_rf | 0.004 [0.001, 0.006] | n/a | 0.008 | 0.99 | 8 |
| B_hgb vs C_slic_400 | 0.164 [0.086, 0.235] | n/a | 0.016 | 1.40 | 8 |
| B_hgb vs F_hybrid | -0.042 [-0.091, -0.005] | n/a | 0.039 | -0.64 | 8 |
| B_rf vs C_slic_400 | 0.161 [0.083, 0.232] | n/a | 0.016 | 1.38 | 8 |
| B_rf vs F_hybrid | -0.046 [-0.092, -0.009] | n/a | 0.023 | -0.71 | 8 |
| C_slic_400 vs F_hybrid | -0.206 [-0.273, -0.138] | n/a | 0.008 | -1.90 | 8 |

> Only 8 paired test images: significance tests are under-powered; report effect sizes and CIs, not 'no difference'.

## Predeclared weighted-rank selection

1. B_hgb (score 0.850)
2. F_hybrid (score 0.830)
3. B_rf (score 0.653)
4. C_slic_400 (score 0.377)
5. A_rgb_euclid (score 0.230)
6. A_lab_de2000 (score 0.010)

Ranking weights are defined in `cucorr.evaluation.statistics.select_best`; a top rank under these weights is not a universal winner - inspect per-constraint columns.
