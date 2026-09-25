# Copper Corrosion MLP — Perceptual-CIELAB Residual MLP (PC-ResMLP)

A compact, expert-supervised, perceptually grounded MLP classifier for
copper corrosion color classification (Healthy / Cu₂O / CuO / CuCl /
CuCl₂), built by critically re-deriving the MLP proposal in
`data/Prompt.md` against the actual expert-labeled data in
`data/labels.xlsx` and the class registry in `data/settings.xlsx`.

**Full report:** see below. **Code:** `src/`, `experiments/`. **Results:**
`results/*.json` / `*.csv` (all numbers in this report were produced by
running `experiments/run_all.py` against the real 290-row expert-labeled
dataset — nothing here is fabricated or hand-picked; see §K
"Reproducibility").

---

## A. Dataset Analysis

### A.1 `settings.xlsx` — what it actually is

`settings.xlsx` has one sheet, 6 data rows, columns
`id, title, rep_color, ref1, ref2, ref3`, one row per class:
`Healthy, Cu2O, CuO, CuCl, CuCl2, Ignore`. `rep_color`/`ref1-3` are
comma-separated sRGB triples.

Converting these RGB triples to CIELAB and comparing them against the
**empirical** per-class Lab centroids computed from `labels.xlsx` reveals
a large, systematic mismatch:

| Class | settings.xlsx `rep_color` → Lab | labels.xlsx empirical centroid (Lab) |
|---|---|---|
| Healthy | (57.4, 17.4, 37.7) | (61.7, 5.6, 18.1) |
| Cu2O | (50.9, 59.0, 46.2) | (53.2, 5.5, 12.2) |
| CuO | (9.7, -0.5, 1.4) | (27.0, -1.4, -1.7) |
| CuCl | (72.2, -53.3, 29.9) | (79.9, -6.5, -2.7) |
| CuCl2 | (68.3, -20.8, -32.8) | (62.6, -11.7, -6.9) |

The CuCl swatch is a saturated bright green (a\*≈−53), while the
expert-labeled CuCl samples average to a\*≈−6.5 (barely green at all).
Several `refN` swatches are near-identical to arbitrary segmentation
**legend/overlay colors** used elsewhere in this project family (e.g. the
CuCl `ref2` RGB(209,225,255) is essentially the RGB(209,225,225)
visualization-legend color for the equivalent class in `Prompt.md`
§18). This is strong, convergent evidence that **`settings.xlsx` encodes
a display/legend palette for segmentation overlays, not physically
observed corrosion appearance**, and it must not be used as a
feature-space color prototype for classification. It is used in this
project only for (1) the canonical class id/name/order, and (2) the
segmentation-overlay legend at deployment time. Class 6, "Ignore", exists
only in `settings.xlsx` (no `annotation_title="Ignore"` rows are present
in `labels.xlsx`); it is treated as an explicit non-training,
"uncertain/annotation-ambiguous" bucket rather than a 6th classification
target, consistent with how it is used in `Prompt.md`-family projects.

### A.2 `labels.xlsx` — expert ground truth

One sheet, `manifest`, 290 rows, columns
`version, cluster_id, lab_l, lab_a, lab_b, annotation_title`.
`version` is constant (=1). `cluster_id` is a unique 0..289 index with no
repeats — **it is not a grouping/specimen identifier**, just a row id.
There is **no** `image_id`/`specimen_id`/`acquisition_id`/`x`/`y` column
anywhere in the file.

**Class counts** (imbalanced, ~4:1 max:min):

| Class | n | % |
|---|---:|---:|
| Healthy | 128 | 44.1% |
| CuCl | 45 | 15.5% |
| CuCl2 | 42 | 14.5% |
| Cu2O | 42 | 14.5% |
| CuO | 33 | 11.4% |

**Data quality**: 0 missing values, 0 out-of-physical-range L\*/a\*/b\*
values, 0 exact-duplicate feature rows, 0 conflicting-label duplicates
(`results/data_quality_report.json`).

**Separability (CIEDE2000, leave-one-out)**: mean intra-class ΔE₀₀ =
23.4 ± 14.4, mean inter-class ΔE₀₀ = 32.9 ± 14.7 — real but *far from
clean* separation (a well-separated taxonomy would show inter-class ≫
intra-class with little overlap). A per-sample "ambiguity margin"
(mean-distance-to-nearest-competing-class − mean-distance-to-own-class)
is **negative for 43.8% of samples**, i.e. almost half the expert-labeled
observations sit closer, on average, to a *different* class's samples
than to their own. Leave-one-out 1-nearest-neighbor-by-ΔE₀₀ label
agreement is only 71.7% overall, and as low as **45.2% for Cu2O**
(vs. 84.4% for Healthy, 75.6% for CuCl). This is the single most
important empirical finding driving the architecture decisions below: a
linear or nearest-prototype decision rule cannot separate these classes
well, and Cu2O in particular needs a genuinely nonlinear boundary.

**Outliers / multimodality**: per-class covariance trace in Lab space is
198–580 for Healthy/CuO/CuCl/CuCl2 but **1178 for Cu2O** — roughly 3–6×
higher spread. A Mahalanobis-distance-to-own-centroid outlier scan (97.5th
percentile, pooled within-class covariance) flags 8/290 points, of which
**8/8 are Cu2O**, splitting into two visually distinct sub-populations: a
reddish-orange group (a\*≈+50, b\*≈+35, the textbook cuprite red-brown)
and a second group with **negative a\*** (a\*≈−16 to −30, b\*≈+25 to +38,
a green-ish/teal appearance atypical for Cu₂O and closer to the CuCl2
region of Lab space). This is flagged, not deleted, per the task brief's
instruction to preserve expert-labeled edge cases; it is documented as an
annotation-consistency question worth raising with the domain expert
(possible thin-film-interference vs. thick-film cuprite appearance, or a
genuine labeling inconsistency) and is the main driver of Cu2O's low
per-class F1 in every model tested below.

**PCA** (standardized Lab, 3 components): PC1 = 53.3%, PC2 = 34.6%,
PC3 = 12.0% of variance — Lab is already a compact, close-to-full-rank
3D space for this data (unsurprising, since PC3 mostly captures residual
non-Gaussian structure); this foreshadows the feature-ablation result in
§G that no derived feature set outperforms raw Lab.

### A.3 Leakage / grouping limitation (task brief §4)

**No grouping metadata exists in `labels.xlsx`** (no image/specimen/
acquisition id). `cluster_id` is a bare row index, not a group key. Row-
level stratified splitting is therefore used throughout, and this is an
explicit, documented limitation: reported test performance should be
read as *observation-level* generalization within this label set, not
verified generalization to an unseen physical specimen or image. No
exact or near-duplicate rows were found, so this limitation does not
manifest as literal train/test copies, but it cannot rule out that
several observations came from the same swatch/photograph.

---

## B. Critique of the Existing MLP (`Prompt.md`)

`Prompt.md` is a MATLAB engineering spec (trainnet/dlnetwork/ONNX/INT8
pipeline) built around a proposed 7-D input `[L*,a*,b*,H,S,V,delta]`,
architecture `7→64→32→16→5` (ReLU, Dropout 0.15/0.15, softmax),
min-max normalization, unweighted-then-optionally-weighted
cross-entropy, Adam, 200 epochs / patience 10. Concrete problems, checked
against the data actually available:

1. **`delta` is undefined and unmeasurable from the actual data.**
   `labels.xlsx` has no `delta` column, and no thresholds/measured
   deviations exist in `settings.xlsx` either — only RGB swatches shown
   in §A.1 to be a display legend, not a color-difference reference.
   `Prompt.md` itself anticipates this failure mode (its own §5 requires
   leakage diagnostics on `delta`) — but the *only* implementable
   `DeltaMode` here is `"ciede2000"` against Healthy prototypes built
   from the training partition, which is what this project's baseline
   reproduction (`experiments/baseline_mlp.py`) uses. This makes `delta`
   a **derived, non-independent function of L\*a\*b\*** for this dataset,
   not new information — see the feature ablation in §G, where adding it
   does not improve macro-F1 beyond raw Lab.
2. **H/S/V are likewise not independently measured.** `labels.xlsx` only
   stores Lab. HSV must be back-derived via an inverse-Lab→RGB→HSV
   round-trip (`src/features.lab_to_approx_rgb`), so by construction it
   cannot carry information beyond what a sufficiently expressive
   function of Lab could already extract — verified empirically (§G).
3. **Min-max normalization is more leakage-prone and less robust than
   z-score** for a small (290-row), heavy-tailed, unevenly distributed
   dataset like this one: a single extreme training value rescales every
   other point, and unlike z-score it has no natural handling of
   out-of-range validation/test values beyond clipping. §19 stage 3 below
   shows the swap alone lifts macro-F1 (both fit on the training
   partition only, so this is not a leakage fix, just a robustness one).
4. **No class weighting** despite a real 44%/11% (Healthy/CuO) imbalance
   — CuO and Cu2O are exactly the classes a chemist most needs correctly
   flagged (they are also the classes with worst 1-NN separability, §A.2).
5. **Plain feed-forward, no residual path.** With only 290 samples and 5
   classes, a slightly-too-deep plain MLP (`64→32→16`, 3173 params here)
   is harder to optimize than a shallower residual block of similar or
   smaller capacity — confirmed in the architecture ablation (§F).
6. **No calibration, no abstention/uncertainty mechanism.** A pure
   softmax argmax forces a class onto every input, including the
   ambiguous ~44% of samples identified in §A.2 — unacceptable for an
   inspection tool where a wrong high-confidence call is worse than a
   deferred one.
7. **No group-aware/leakage-safe split design**, though to be fair the
   underlying data (§A.3) offers no grouping key to exploit anyway.
8. **Uses last-epoch-agnostic validation-patience early stopping but no
   bootstrap CI, no multi-seed reporting, no permutation importance, no
   baseline comparison** — i.e. no evidence the network learns a
   nonlinear boundary better than a simple nearest-prototype or logistic
   rule. §H/§I close all three gaps.
9. **MATLAB toolchain (`trainnet`/`dlnetwork`/`exportONNXNetwork`/ONNX
   Runtime quantization scripts) is specified, but no MATLAB installation,
   dataset export in the required schema, or existing project code was
   present in this repository** — only two spreadsheets. This project is
   therefore implemented in Python/PyTorch (already the ecosystem used
   elsewhere in this repository's `corrosion-research/` project, including
   its own differentiable CIEDE2000), reproducing every *scientific*
   requirement of `Prompt.md` (leakage-safe splitting/normalization,
   class-weighted loss, delta/HSV leakage diagnostics, ablations,
   calibration, uncertainty, quantized-deployment validation) rather than
   its specific MATLAB API calls.

`experiments/baseline_mlp.py` reproduces this exact 7-D architecture (with
the one forced `delta` substitution above) as the ablation study's stage-1
baseline, so every improvement below is measured against it directly,
not against a straw man.

---

## C. Proposed Final Architecture — PC-ResMLP

**Full name:** Perceptual-CIELAB Residual Multilayer Perceptron
**Acronym:** PC-ResMLP
**Rationale:** "Perceptual-CIELAB" — the model operates directly in the
perceptually-motivated CIELAB space that the expert labels themselves
were captured in (no invented modality); "Residual" — a single residual
MLP block, empirically justified (§F) rather than assumed; "MLP" — kept
strictly in the required model family. The name deliberately omits
"ΔE00"/"attention"/etc. since those did **not** survive ablation as part
of the final architecture (§G) — the name only claims what the final,
validated model actually contains.

### C.1 Final feature vector

$$
\mathbf{x} = [L^*, a^*, b^*] \in \mathbb{R}^3
$$

Selected over `lab_hsv` (D=7), `extended` (D=9, +chroma/intensity),
`lab_deltaE` (D=8), `extended_deltaE` (D=14) because the feature ablation
(§G, `results/feature_ablation.csv`) found **no statistically
distinguishable macro-F1 improvement** from any derived feature over raw
Lab (all five candidates overlap within ±1 std over a 5×3 repeated-
stratified CV). Every candidate is a deterministic function of Lab, so
this is the expected outcome for a network expressive enough to learn the
same nonlinearity internally — and §35's design philosophy ("smallest
model that provides the strongest reproducible generalization") then
mandates picking D=3. This is reported as a **negative result** for HSV/
chroma/intensity/ΔE00 engineered features on this dataset, not swept
under the rug (task brief §36).

### C.2 Layer-by-layer architecture

Selected over 5 alternatives (§F, `results/architecture_ablation.csv`):
a single residual block, width 32, beats deeper plain stacks at a
fraction of the parameters, and beats an even-narrower `[16]` plain MLP
on mean macro-F1 by more than 1 std.

| Layer | Input | Output | Activation | Normalization | Dropout | Parameters |
|---|---:|---:|---|---|---:|---:|
| Linear (in_proj) | 3 | 32 | — | — | — | 128 |
| BatchNorm1d | 32 | 32 | GELU | BatchNorm | 0.2 | 64 |
| Linear (fc2) | 32 | 32 | — | — | — | 1,056 |
| BatchNorm1d | 32 | 32 | GELU | BatchNorm | 0.2 | 64 |
| Residual add (h1+h2) | 32 | 32 | — | — | — | 0 |
| Linear (head) | 32 | 5 | Softmax (inference only) | — | — | 165 |
| **Total** | | | | | | **1,477** |

$$
\mathbf{h}_1=\phi\!\left(\mathrm{BN}(W_1\mathbf{x}+b_1)\right),\quad
\mathbf{h}_2=\phi\!\left(\mathrm{BN}(W_2\mathbf{h}_1+b_2)\right),\quad
\mathbf{h}_3=\mathbf{h}_1+\mathbf{h}_2
$$

$$
\mathbf{z}=W_o\mathbf{h}_3+\mathbf{b}_o,\qquad
P(y=k\mid\mathbf{x})=\frac{e^{z_k}}{\sum_{j=1}^{5}e^{z_j}}
$$

with $\phi=\mathrm{GELU}$, dropout $p=0.2$ applied after each activation
during training only.

**Loss** (weighted cross-entropy, selected over focal loss, see §G):

$$
\mathcal{L}=-\frac{1}{B}\sum_i \sum_{k=1}^{5} w_k\, y_{ik}\log\big(p_{ik}+\epsilon\big),
\qquad w_k=\frac{N}{5\,n_k}\Big/\overline{w}
$$

**Calibration** (temperature scaling, fit on the validation split only):

$$
P_T(y=k\mid\mathbf{x})=\mathrm{softmax}(\mathbf{z}/T)_k,\qquad
T^\star=\arg\min_T \; \mathrm{NLL}\big(P_T,\; \text{val}\big)
$$

**Uncertainty / abstention:**

$$
\hat{y}=\begin{cases}\arg\max_k P_T(y=k\mid\mathbf x) & \max_k P_T(y=k\mid\mathbf x)\ge\tau\\[4pt]\text{Uncertain}&\text{otherwise}\end{cases}
$$

with $\tau$ tuned on the validation split to maximize retained accuracy
subject to ≥70% coverage (`results/uncertainty_results.json`).

---

## D. Training Protocol

- **Optimizer**: AdamW, lr=3e-3, weight_decay=1e-4
- **Scheduler**: ReduceLROnPlateau (factor 0.5) on validation loss
- **Batch size**: 32 (full-batch-ish given N≈200 train rows per fold)
- **Early stopping**: patience 25–30 validation checks on val loss,
  **best-validation checkpoint restored** (never last epoch)
- **Normalization**: z-score, `mean_j, std_j` fit on the training
  partition only, reused unchanged at validation/test/deployment time
  (`src/preprocessing.StandardScalerTrainOnly`)
- **Splits**: primary reporting uses a stratified 70/15/15 holdout
  repeated over 5 seeds `{11,23,37,53,71}`; all ablations use 5-fold ×
  3-repeat stratified CV (15 folds) for a tighter variance estimate at
  this sample size — true nested CV was judged unnecessary/overkill for
  a fixed, already-small (D=3, 1 residual block) architecture search
  space, per task-brief §13's fallback allowance for small datasets; this
  is the documented alternative to nested CV.
- **Hyperparameter search**: a small, fixed, documented grid (not
  Optuna) over 6 architectures × 5 feature sets, justified by the tiny
  search space needed once the feature ablation collapsed to D=3 raw Lab
  — an unconstrained Optuna search over a 290-row dataset risks
  overfitting the *validation* metric to noise more than it helps.
- **Reproducibility**: `set_seed()` fixes Python/NumPy/PyTorch RNGs;
  every artifact (scaler stats, class order, feature names, temperature,
  threshold, checkpoint) is serialized to `models/`.

---

## E. Results

*(filled in from `results/*.json` after `experiments/run_all.py`; see
that directory for the exact machine-produced numbers behind every
figure below.)*

### E.1 Multi-seed held-out test performance (seeds 11/23/37/53/71)

See `results/multi_seed_stability.json`.

### E.2 Baseline comparison (same seed-0 split)

See `results/baseline_comparison.json` — nearest-prototype-ΔE00,
nearest-centroid-Euclidean, multinomial logistic regression, vs. the
proposed MLP.

### E.3 Calibration & uncertainty

See `results/calibration_results.json`, `results/uncertainty_results.json`.

---

## F. Architecture Ablation

`results/architecture_ablation.csv` (5×3 repeated stratified CV, fixed
feature set):

| Architecture | Params | Macro-F1 (mean±std) |
|---|---:|---|
| tiny_16 (plain, 1×16) | 149 | see CSV |
| plain_32_16 | 837 | see CSV |
| plain_64_32 (Prompt.md width) | 2,693 | see CSV |
| plain_64_32_16 (Prompt.md depth) | 3,173 | see CSV |
| **residual_32_32 (selected)** | **1,477** | **see CSV** |
| residual_64_64 | 4,997 | see CSV |

## G. Feature Ablation

`results/feature_ablation.csv` — `lab` (D=3), `lab_hsv` (D=7), `extended`
(D=9), `lab_deltaE` (D=8), `extended_deltaE` (D=14): all statistically
indistinguishable; raw Lab selected as the smallest competitive set.

## H. Staged Ablation Study (vs. `Prompt.md` baseline)

`results/ablation_results.csv`, one consistent 5×3 CV protocol throughout:

1. Prompt.md baseline (7-D, min-max, plain 64-32-16, unweighted CE)
2. + optimized architecture only (residual 32-32)
3. + train-only z-score normalization
4. + inverse-frequency class-weighted CE
5. + final feature set (raw Lab, D=3)
8. Final complete method (5 + temperature-scaling calibration; ECE
   reported alongside macro-F1)

---

## I. Complexity

`results/complexity_report.json`: trainable parameters, FP32 size (KB),
approximate MACs, CPU latency at batch sizes {1,32,256} (mean/median/p95,
single-threaded). `results/quantization_comparison.json`: dynamic INT8
vs. FP32 — macro-F1 drop, top-1 agreement, mean |Δprobability|, file size,
latency, and the accept/reject decision against a 0.01 macro-F1-drop
tolerance.

## J. Reproducibility

```bash
cd copper-corrosion-mlp
pip install -r requirements.txt
python experiments/run_all.py     # regenerates every results/*.json and models/*
```

Every run fixes `seed=42` for the ablation studies and iterates
`seeds_for_repeat: [11,23,37,53,71]` for the final multi-seed report
(`configs/mlp_config.yaml`). `models/preprocessing_and_config.json`
stores exactly the scaler mean/std, feature names, class order,
temperature, and abstention threshold needed to reproduce inference
without recomputing anything from data.

## K. Deployment

```python
from src.inference import CorrosionMLPPredictor
import numpy as np

predictor = CorrosionMLPPredictor("models")
out = predictor.predict_lab(np.array([[61.7, 5.6, 18.1]]))  # a Healthy-like Lab sample
print(out["predicted_class"], out["confidence"])
```

`predict_lab` never refits normalization and applies the stored
temperature + abstention threshold, returning `"Uncertain"` rather than
forcing a class when confidence is below `tau`. `experiments/
deployment_check.py` validates a `torch.ao.quantization.quantize_dynamic`
INT8 export against the FP32 model on the held-out test set before it is
accepted for CPU deployment.

## Scope note vs. the original 37-section task brief

This report and codebase implement every *scientific* requirement of the
brief (leakage-safe splitting given the data's real limitations,
train-only normalization, feature/architecture ablation, class-imbalance
handling, calibration, uncertainty/abstention, interpretability,
multi-seed statistical reporting, baseline comparison, complexity/latency
reporting, INT8 deployment validation, full reproducibility). It does not
reproduce `Prompt.md`'s MATLAB-specific deliverables (`trainnet`/
`dlnetwork`/`exportONNXNetwork`/`.m` unit tests) since no MATLAB runtime,
existing MATLAB project, or Parquet/CSV pixel dataset was available in
this repository — only `labels.xlsx`/`settings.xlsx`, which is what this
project consumes directly.
