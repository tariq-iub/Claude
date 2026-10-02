# Ablation plan

Implemented arms (`experiments/ablation.py`, run with `scripts/run_ablation.py`; all report per-seed rows):

**Cumulative** (each adds one group on top of the previous): `rgb_only` → `+lab` → `+hsv` → `+texture` → `+specular_diffuse` → `+virtual_stokes` (DoLP_hat, cos/sin 2AoLP_hat) → `+virtual_analyzer` (analyzer images) → `+geometry` (normals, normal discontinuity) → `+roughness` → `+psrf` → `+uncertainty_feats` (latent u, glare g) → `+cartridge_prior`.
**Leave-one-out** from the full model: − specdiff, − vstokes, − analyzer, − geometry, − roughness, − psrf, − uncertainty, − texture, − lab, − cartridge_prior.
**Colour-space arms**: rgb, lab, hsv, rgb+lab, rgb+hsv, rgb+lab+hsv. **Fusion arms**: early, mid, late.
**Training options** (not architecture flags): distillation (needs teacher), synthetic pre-training, residual learned polarization (`residual_gain > 0`), unconstrained image-to-image analyzer head (for H2), single-θ vs PSRF (H5), channel-wise vs single-index Fresnel, environment vs directional illumination prior.

Table `ablation` columns: arm, seed, miou, mdice, mf1, mbalanced_accuracy, severity_mae, params, distilled, data. Interpretation rules: effects judged by paired differences across seeds *and* cartridge-grouped bootstrap, not by single-run rankings; parameter count reported because later arms are larger (add a capacity-matched RGB-only arm: `width_mult` scaled to equal parameters).
