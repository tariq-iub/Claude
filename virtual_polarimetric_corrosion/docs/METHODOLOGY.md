# Methodology

**Key limitation, stated as the research question.** One RGB observation does not uniquely determine the polarization state. *Virtual polarimetry = physics-constrained estimation of plausible polarization-dependent surface responses.* The thesis measures how useful (and how accurate) that estimate is.

## Pipeline
1. **Radiometric linearization** (sRGB → linear; clipping/glare masks). Unknown white balance/tone-mapping are randomised in synthetic pre-training and reported as a limitation.
2. **Geometry prior**: silhouette → cylinder axis, radius → normals (orthographic approximation; axis elevation ignored at test time). Generic (no prior) vs cartridge-aware model is hypothesis H4.
3. **Diffuse–specular separation** with metal-tinted specular colour; neither component assumed (un)polarized.
4. **Latent optical state** Z (bounded proxies + log-variances) → **physics layer** Ŝ.
5. **Virtual analyzer / counterfactual stack / PSRF / analyzer-angle optimisation** (J(θ) = αT + βC + γE − δG − λU, periodic PCHIP, θ*).
6. **Feature fusion** (colour spaces RGB/CIELAB/HSV ± texture ± polarization features; early/mid/late) → **VP-CorrosionNet** → segmentation, PittingHead, severity (ordinal + area %), uncertainty.
7. **CPE**: K plausible latent states → mean/variance/intervals; test whether corrosion evidence is stable across states.
8. **Hardware teacher → RGB student** (optional training stage).

## Three configurations compared
A physical polarization (real analyzer stack); B RGB virtual polarization; C distilled virtual polarization (train with A, deploy RGB-only).

## Tasks
T1 healthy vs corroded; T2 multi-class segmentation; T3 discoloration; T4 pitting (own head); T5 severity (ordinal); T6 corroded-area %; T7 uncertainty/abstention.

## Colour spaces
CIELAB/HSV are inputs *and* an experimental factor: rgb, lab, hsv, rgb+lab, rgb+hsv, rgb+lab+hsv × fusion {early, mid, late}. Hue is circular → (cosH, sinH). ΔE00 to a healthy reference is an auxiliary feature, never chemical evidence.

## Pre-registered decisions (before touching real test data)
* Primary metric for T2: mIoU over classes present in the test set, per-cartridge-group bootstrap CI.
* Severity bins in `configs/default.yaml` are placeholders to be fixed with domain experts first.
* All hyper-parameters tuned on validation groups only; the test split is opened once per final model.
* Negative results are reported with the same prominence as positive ones.
