# Experimental protocol

## Research questions and hypotheses (falsifiable; outcomes not presupposed)
| ID | Question | Hypothesis | Falsified if |
|---|---|---|---|
| RQ1/H0 | Does the physics-constrained model generate useful analyzer-dependent representations from RGB? | Virtual I_θ/DoLP_hat/AoLP_hat beat trivial baselines (constant DoLP, geometry-only, physics-only untrained) against *measured* values | not better than baselines on held-out cartridges (G3) |
| RQ2/H1 | Do virtual-polarization features improve corrosion segmentation under strong glare vs RGB-only (incl. Lab/HSV)? | mIoU(+virtual pol) − mIoU(RGB+Lab/HSV) > 0 in the high-glare stratum | paired CI includes 0 or is negative |
| RQ3 | How closely does RGB-only virtual polarization approximate real? | report image RMSE/SSIM, DoLP MAE, wrapped AoLP error, glare suppression vs hardware | — (measurement) |
| RQ4/H3 | Does distillation reduce the physical–virtual gap? | C closer to A than B is (segmentation + polarization fidelity) | C ≈ B within CI |
| RQ5 | Which optical variables matter? | leave-one-out + cumulative ablation + permutation importance | — |
| RQ6 | Does uncertainty track reconstruction ambiguity? | Spearman(CPE std, |Î_θ − I_θ,measured|) > 0 and abstention reduces risk | ρ ≤ 0 |
| RQ7/H4 | Does cartridge geometry help? | cartridge-prior arm > generic arm on curved cartridges | no gain |
| H2 | Physics-constrained generalises better under illumination shifts than unconstrained image-to-image | smaller mIoU drop under illumination/colour-temperature/exposure shifts | not smaller |
| H5 | PSRF more robust than one selected analyzer angle | PSRF arm degrades less across perturbations than single-θ arm | no difference |

## Data splits (see DATASET_PROTOCOL.md)
Group (cartridge) level: 70/15/15 group-stratified by severity bin; alternative grouped k-fold if < ~60 cartridges. Never split pixels or views of one cartridge across splits (`splits.assert_no_group_leakage`, `scripts/check_leakage.py`). Sessions/cameras held out in external validation.

## Baselines (chosen per question, not all everywhere)
* Image operators: original RGB, CLAHE, gamma, MSR Retinex, highlight suppression, specular removal; classical: CIELAB (CIEDE2000+Otsu) and HSV thresholding.
* Learned RGB: UNetLite, mobile RGB-only (same backbone). **DeepLabV3+ and YOLO**: implement an adapter returning `{'seg_logits': (B,K,H,W)}` / boxes with the same `evaluate()` contract; not bundled to keep dependencies minimal.
* Polarization: simple pseudo-polarizer (strawman), physics-only virtual polarizer, VP-CorrosionNet, distilled VP-CorrosionNet, (optional) GenPolar/Lin et al. models as RGB→Stokes baselines if code exists.

## Metrics
Segmentation: IoU/mIoU, Dice, precision, recall, specificity, F1, balanced accuracy (per class + macro over present classes). Detection (pits/boxes): mAP50, mAP50–95, P, R. Severity: MAE, RMSE, R², QWK, off-by-one accuracy, Spearman. Polarization quality: glare suppression ratio, highlight-area reduction, edge/texture/gradient preservation, local contrast, entropy, signal-to-glare ratio (grayscale, cartridge pixels, exposure-matched). Agreement with measurement: I_θ RMSE/PSNR/SSIM, DoLP MAE/RMSE, **wrapped AoLP error (DoLP-weighted)**, cos2Δφ agreement. Calibration: ECE, Brier, NLL, reliability diagrams, risk–coverage/AURC. Runtime: latency (mean/median/p95), FPS, RAM, VRAM, parameters, size, MACs (CPU, GPU→CPU fallback, optional ONNX Runtime).

## Robustness
Illumination intensity, colour temperature, exposure (stops, with clipping), white balance, glare blob, rotation, blur, noise, camera shift, background replacement, JPEG, curvature (radius). Degradation curves = *measured points* (no smoothing) over ≥ 5 levels, ≥ 3 seeds.

## Hardware-to-software gap (central experiment)
Samples with real stacks: compare physical analyzer image vs virtual vs distilled-virtual for: image similarity, polarization-feature agreement, glare suppression, segmentation, classification, latency (`experiments/hardware_gap.py`). Pairing by cartridge group.

## Synthetic data (separate, never mixed into real tables)
`experiments/synthetic.py` + `scripts/run_synthetic_validation.py`: unit-validation of physics modules and domain-randomised pre-training. Table column `data_origin` ∈ {SYNTHETIC, REAL}. Sim-to-real benefit tested by: train on synthetic → fine-tune real vs real-only (same seeds).

## Seeds and reporting
≥ 5 seeds when feasible (≥ 3 minimum). Report mean ± SD and 95 % t-CI over seeds *and* grouped bootstrap over test cartridges. See STATISTICAL_ANALYSIS.md.
