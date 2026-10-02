# Threats to validity

**Internal.** (i) Leakage: multiple views/frames of one cartridge across splits → mitigated by group splits + checker; near-duplicates across groups → embedding check. (ii) Tuning on test: hyper-parameters, thresholds (severity, abstention, θ* weights) chosen on validation groups only. (iii) Seed variance: ≥ 3 seeds, paired designs. (iv) Capacity confound: physics arms have more parameters than RGB-only → capacity-matched control. (v) Metric artefacts: classes absent in test excluded from macro means (reported); AoLP evaluated with wrapped, DoLP-weighted error; glare metrics computed on cartridge pixels with exposure matching (an earlier whole-frame version of Fig. 6 was misleading because background saturation dominated — fixed). (vi) Annotation noise: two annotators, κ, ignore bands. (vii) Exposure confound in hardware comparisons: analyzer images pass ≈ half the light; compare at matched exposure policy. (viii) Teacher-to-student leakage via S0-from-stack inputs (use real RGB captures).

**Construct.** "Polarization" in virtual outputs is an estimate; agreement with hardware must be shown, not asserted. Corrosion "subtypes" are visual labels. Severity bins are conventions. Glare robustness defined by saturated-pixel fraction strata — report alternatives.

**External.** One camera/lighting rig/site/alloy lot → external set with new camera, session, lot, finish. Synthetic-to-real: renderer shares the Fresnel/microfacet model with the estimator, so synthetic success is optimistic by construction (circularity) — synthetic results are for code validation and pre-training only. Cartridge geometry prior may not transfer to other objects.

**Statistical conclusion.** Few independent units (cartridges) → wide CIs; group bootstrap; no significance claims from n < 6 pairs; multiplicity control; effect sizes reported; negative results reported.

**Novelty/literature.** Partial search, snippet-level verification; closest prior art (IEVPF, Lin et al., GenPolar) unread in full → novelty claims are provisional.
