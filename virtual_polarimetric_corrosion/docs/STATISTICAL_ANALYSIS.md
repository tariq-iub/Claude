# Statistical analysis plan

* **Units.** Independent unit = cartridge group (not image, not pixel). Seeds are repeats of training, not independent samples of data.
* **Reporting.** Per metric: mean ± SD over seeds (≥ 3, target 5) and 95 % t-interval (`stats.mean_sd_ci`); cartridge-clustered bootstrap CI (`grouped_bootstrap_ci`, resample groups); paired bootstrap of differences (`paired_bootstrap_diff`).
* **Comparisons.** Pre-register the comparison family before test evaluation (e.g., primary: +virtual-pol vs RGB+Lab/HSV on mIoU in the high-glare stratum; B vs C; A vs C). Paired design over cartridges. Wilcoxon signed-rank only when n ≥ 6 pairs (below that the exact test cannot reach p < 0.05 and the code returns NaN rather than pretending); paired t reported alongside for reference.
* **Effect sizes.** Cohen's d_z (paired), Cliff's δ (ordinal/non-normal); interpret together with CIs. Practical-significance threshold fixed in advance (e.g., ΔmIoU ≥ 0.02).
* **Multiplicity.** Holm–Bonferroni within a family (`holm_bonferroni`); Benjamini–Hochberg for exploratory ablation screens (`benjamini_hochberg`), labelled exploratory.
* **No mechanical significance testing.** Tests are used only for the pre-registered family; ablation rankings are descriptive with CIs.
* **Circular statistics.** AoLP errors use wrapped distance (period π); mean AoLP via circular mean of e^{i2φ}; never arithmetic mean/difference of raw angles.
* **Calibration.** ECE (15 bins; also report bin-count-robust alternative), Brier, NLL; compare before/after with paired bootstrap over cartridges.
* **Multiple seeds for synthetic runs** are treated separately (data_origin = SYNTHETIC) and never pooled with real results.
* **Curves.** Measured points plotted; fitted curves (PCHIP/LOESS/Savitzky–Golay/spline/regression) only where justified, with raw and fitted CSVs exported (`visualization/curves.py`).
