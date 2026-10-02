# Hardware-teacher → software-student distillation

**Claim to be tested, not assumed:** supervising an RGB-only student with measured polarimetric targets improves RGB-only deployment ("hardware-supervised, hardware-free").

## Data
Hardware subset with analyzer stacks (0/45/90/135 minimum). Student input: the *RGB image of the same pose*. Use the unpolarized RGB from a separate no-analyzer capture, or S0 from the stack — state which (S0-from-stack is a domain shift from real RGB: the student must be evaluated on genuine RGB captures).

## Stages
1. **Teacher** (`PolarimetricTeacher`): input stack → measured Stokes (least squares) → fused features → corrosion outputs. Train on train groups; early-stop on val groups; report teacher test accuracy (configuration A).
2. **Student baseline B**: VPCorrosionNet trained without teacher.
3. **Distilled student C**: same net, loss = L_task + λ_distill·L_distill with L_distill = circular Stokes loss (Ŝ vs measured S) + feature MSE (projection of student decoder features vs teacher features) + temperature KL on logits (T = 2). Teacher frozen.
4. Optional **staged** variants: (i) Stokes-only distillation (representation), (ii) logits-only, (iii) all — to attribute gains.
5. Optional pre-training on synthetic (domain randomised) then distill.

## Evaluation (hardware-to-software gap)
Per held-out cartridge: segmentation (mIoU, per-class), severity, polarization fidelity (DoLP MAE, wrapped AoLP error, I_θ SSIM), glare suppression, CPU latency, for A, B, C. Paired comparisons over cartridges (Wilcoxon where n ≥ 6, paired bootstrap, Cohen's d_z, Holm correction across the pre-registered comparison family: B vs C, A vs C, A vs B on the primary metrics). "Gap closed" = (C − B)/(A − B) with a bootstrap CI — undefined/uninformative when A ≈ B; report raw values.

## Failure modes to check
Teacher overfitting small data (compare teacher val vs test); label leakage through shared groups; student matching S0 trivially while AoLP stays random (inspect wrapped AoLP error); distillation hurting calibration (ECE before/after); gains only on the hardware-subset cartridges (check generalisation to non-hardware cartridges).

## Decision rule
If C ≤ B within CI on all primary metrics, conclude *no demonstrated benefit at this data scale* and report the measured gap; do not tune hyper-parameters on the test groups to rescue the claim.
