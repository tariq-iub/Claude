# Limitations (known, structural, and current)

## Structural (will remain)
1. **A single RGB image does not determine the polarization state.** The latent optical state is under-determined (≥ 10 unknowns vs 3 observations per pixel). Virtual polarimetry is an estimate conditional on priors; the CPE shows alternatives, it does not remove the ambiguity.
2. **Diffuse/specular attribution dominates the AoLP error.** Measured on 32 synthetic views (`docs/SYNTHETIC_VALIDATION_REPORT.csv`, SYNTHETIC): in environment-illumination views (where the baseline's illumination assumption holds) the untrained physics-only baseline has mean wrapped AoLP error 71° with cos2Δφ agreement −0.63 — a systematic near-90° flip, worse than the 45° chance level — whereas an oracle given the *true* diffuse/specular split and true roughness (same cylinder-prior normals) reaches 9° (cos2Δφ 0.87). The Fresnel/Stokes algebra is therefore not the bottleneck; the split (and possibly roughness — the oracle supplies both, so they are not separated here) is. In directional-light views the same baseline gets 14° (oracle 10°), so conclusions depend on the illumination regime. Caveat: renderer and estimator share one physics model, so the oracle is an upper bound, not a real-data result.
3. **Copper's visible-range polarization is weak** in the red channel (high k); most usable signal is expected in G/B, at the cartridge limb, and from corrosion layers. If measured DoLP of corroded copper is small relative to sensor noise, virtual polarization cannot add information — an empirical question for the hardware rig.
4. **Camera model unknown.** Spectral response, white balance and tone-mapping are unknown; RGB-effective optical constants are an approximation; clipping destroys information that no model recovers.
5. **Microfacet/Fresnel validity.** Geometric optics, single scattering, semi-infinite layers; thin-film interference, volume scattering (powdery chloride/patina), pits near the masking scale and anisotropic finishes (brushing, lathe marks) are outside the model.
6. **Geometry.** Cylinder prior is orthographic, axis in the image plane (elevation ignored at test time), no ogive/rim/headstamp; real cartridges deviate. Monocular normal estimation is ambiguous.
7. **Annotation is visual.** Classes are "visually annotated corrosion classes"; no chemical identification is claimed (needs XRF/Raman/XRD ground truth).

## Current (this repository)
* **No real data, no real polarization measurements, no trained-model results.** All numbers produced here are from synthetic data and are labelled SYNTHETIC; they validate code, not the science.
* Optical constants are proxies; bronze/brass = midpoint of Cu and Au tables (assumption).
* **Open physics issue:** source/analyzer frame convention for the cross-polarized chain (POLARIZATION_PHYSICS §7). Not used by the network.
* The roughness depolarization r0 and the environment-illumination mode are modelling assumptions awaiting hardware validation.
* DeepLabV3+ and YOLO baselines not bundled; RGB→polarization competitors (Lin et al.; GenPolar) not run.
* Bibliography not Crossref-verified (egress blocked); IEVPF paper unread; the primary paper's full text unread (abstract-level only).
* Severity bins are placeholders. Group-level splitting needs enough cartridges (target ≥ 60).
* Efficiency numbers (parameters/latency) are from a development container, not target edge hardware.
