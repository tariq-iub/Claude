# Research gap

1. **Hardware dependence of polarization inspection.** Published polarization-based defect detectors (e.g., Yu et al. 2025) need polarization optics at inference. Deployed inspection stations, handheld cameras and archived photographs do not have them.
2. **RGB→polarization exists, but not for inspection of metals.** Benchmarks/generative models (Lin et al. 2025; GenPolar) estimate polarization from RGB for general scenes. They do not model conductor reflection, curved-cartridge geometry, or corrosion layers, and do not test whether the estimate helps an inspection decision.
3. **Ambiguity is rarely propagated.** Single-image polarization has structural ambiguities (diffuse vs specular π/2 flip, π azimuth ambiguity, roughness/index/normal confounding). Existing estimators return one answer. A counterfactual ensemble that is evaluated for *corrosion-evidence stability* is not found in the reviewed work.
4. **Corrosion vision ignores optics.** Corrosion segmentation is appearance-only RGB; glare and metal reflection are treated as noise/augmentation rather than modelled.
5. **Missing quantification.** Nobody (in the reviewed set) reports the *hardware-to-software gap*: physical analyzer vs virtual analyzer vs distilled virtual analyzer on the same samples for the same downstream task.
6. **No cartridge corrosion benchmark** (none found; coverage limited).

**Research gap statement.** There is no established, validated way to obtain polarization-conditioned corrosion evidence from ordinary RGB images of curved copper/bronze surfaces, and no measurement of how much of the benefit of physical polarization imaging such a software-defined camera can recover. This thesis turns that into the question in the README and tests it with hypotheses H1–H5 (`EXPERIMENTAL_PROTOCOL.md`).
