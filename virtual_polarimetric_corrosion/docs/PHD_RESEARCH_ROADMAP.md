# PhD research roadmap

**Thesis question.** To what extent can the useful corrosion-discriminative information produced by physical polarization imaging be
transferred into a physics-constrained, software-defined virtual polarimetric camera operating from ordinary RGB images?

Time scale below assumes a 4-year programme (months in brackets are indicative). Each phase ends with a **gate**; failing a gate changes
the plan (see "Pivot rules"), it does not get re-worded into a success.

## Phase 1 — Literature review (M1–6, continuing)
* **Objective** map polarization imaging, reflection separation, SfP, RGB→polarization estimation, metal inspection, corrosion vision.
* **Tasks** run the search protocol in `LITERATURE_REVIEW.md` §1 (databases, strings, inclusion/exclusion, dated log); read the two near
  neighbours in full (IEVPF virtual polarization filtering; Lin et al. RGB-to-Polarization); run `scripts/verify_bibliography.py`.
* **Inputs** `BIBLIOGRAPHY.csv`, `NOVELTY_MATRIX.csv`. **Outputs** verified bibliography, updated novelty matrix, gap statement.
* **Risks** prior art invalidates a claimed contribution (already partly true: RGB→polarization is not new). **Validation** every
  citation Crossref-matched; novelty matrix re-reviewed by a second reader.
* **Gate G1** the claims in `NOVELTY_ANALYSIS.md` still hold after reading IEVPF and Lin et al. in full.

## Phase 2 — Physical polarization theory (M2–8)
* Derivations in `MATHEMATICAL_FORMULATION.md`; resolve the **source/analyzer frame-convention** open issue
  (`POLARIZATION_PHYSICS.md` §7) analytically and on a bench mirror. Inputs: textbooks, Chipman et al. Outputs: verified Mueller chain,
  unit tests. Risk: sign-convention errors. Validation: extinction/Malus/Brewster tests + bench mirror + dielectric reference.

## Phase 3 — Real polarization imaging prototype (M6–14)
* Build the rig of `HARDWARE_ACQUISITION_PROTOCOL.md` (linear camera, rotating polarizer on a calibrated mount, ring/softbox sources,
  cylinder rotation stage). Calibrate polarizer zero, extinction ratio, flat-field, sensor linearity. Inputs: hardware budget.
  Outputs: calibration report; measured Stokes of reference targets (dielectric, polished copper). Risks: polarizer leakage, light drift,
  registration. **Validation:** measured DoLP/AoLP of a dielectric plate vs Fresnel prediction within calibrated uncertainty;
  fit residual of the Malus model (`fit_rmse`) at noise level. **Gate G2** the rig reproduces known Fresnel behaviour.

## Phase 4 — Dataset acquisition (M10–22)
* `DATASET_PROTOCOL.md`: spent/inert cartridges, stratified corrosion levels/subtypes, ≥3 acquisition sessions, group IDs, expert
  annotation (two annotators + adjudication, κ reported), hardware-teacher subset with 4 (and 8) angles. Outputs: manifest CSV, splits
  by cartridge group, annotation protocol, datasheet. Risks: too few cartridges for group-level splits (need ≥ ~60 for 15 % test),
  annotation noise. **Validation:** inter-annotator agreement; leakage checker passes.

## Phase 5 — Physics-based simulator (M8–14, parallel)
* `experiments/synthetic.py` (exists) → extend with real optical constants, measured pBRDF (e.g., Baek et al. dataset) for metals,
  measured camera spectral response. Outputs: domain-randomised generator. Risk: sim-to-real gap. Validation: simulator reproduces the
  Phase-3 reference measurements.

## Phase 6 — RGB virtual polarizer (M12–20)
* Physics-only baseline → learned latent estimator; evaluate against **measured** analyzer images/DoLP/AoLP (wrapped AoLP error!).
  Outputs: `VirtualAnalyzer` + latent estimator validated vs hardware. **Gate G3** virtual DoLP/AoLP beat the trivial baselines
  (constant-DoLP, geometry-only) on held-out cartridges; otherwise pivot (below).

## Phase 7 — VP-CorrosionNet (M16–26)
* Train/ablate (cumulative, leave-one-out, colour space, fusion). Outputs: model, ablation tables with seeds, CIs. Validation:
  protocol in `EXPERIMENTAL_PROTOCOL.md`; hypotheses H1–H5 tested as written.

## Phase 8 — Hardware→software distillation (M22–30)
* `DISTILLATION_PROTOCOL.md`: teacher on stacks; student with L_distill; configurations A/B/C. Output: the hardware-gap experiment.
  Validation: paired comparison over test cartridges, effect sizes, Holm correction. Risk: teacher overfits small data.

## Phase 9 — Uncertainty (M26–32)
* Heteroscedastic + MC dropout + ensembles + CPE; ECE/Brier/NLL/risk–coverage; test RQ6 (does uncertainty track reconstruction ambiguity).
  Validation: calibration on held-out cartridges; abstention improves risk at fixed coverage.

## Phase 10 — Ablation & benchmarking (M28–36)
* Full tables, robustness curves, runtime on CPU / low-end GPU / ONNX Runtime. Outputs: all CSVs from `tables/` schemas filled from runs.

## Phase 11 — External validation (M32–40)
* New camera, new site/session, new cartridge lot, unseen finish. Validation: performance drop reported, not hidden.

## Phase 12 — Thesis & publications (M36–48)
* Papers (see `PAPER_DRAFT_MATERIAL.md`): P1 virtual polarimetry from RGB; P2 polarization-aware corrosion segmentation; P3 distillation
  (hardware→software); P4 uncertainty; P5 edge deployment. Merge P4/P5 if results are thin — do not fragment.

## Pivot rules (decided in advance)
| If | Then |
|---|---|
| G3 fails (virtual AoLP no better than geometry-only baseline) | Drop AoLP from the claim; keep DoLP/analyzer-conditioned *features*; reframe contribution as physics-regularised glare-robust features |
| Virtual polarization gives no corrosion gain over RGB+Lab/HSV (H1 rejected) | Report as negative result; the hardware-teacher distillation may still help representation (test H3 separately) |
| Distilled student ≈ RGB student (H3 rejected) | Conclude hardware is not transferable at this data scale; quantify the gap |
| Too few cartridges for group splits | Use grouped cross-validation with fixed folds; lower claims accordingly |
