# Novelty analysis (formal review; claims deliberately conservative)

Method: compare each candidate contribution against the matrix `NOVELTY_MATRIX.csv` (16 entries; 8 at `verified_search`, the rest partial).
Limits: partial literature coverage (see `LITERATURE_REVIEW.md` §1). **No "first" claim is made anywhere.** Any claim below is conditional on
being implemented *and* supported by experiments (none are yet).

| # | Candidate contribution | Overlap found | Verdict |
|---|---|---|---|
| 1 | Physics-constrained virtual polarimetric camera | IEVPF ("virtual polarization filtering", unread); RGB→polarization (Lin 2025, GenPolar) | **Not novel as a concept.** Defensible differentiator only if: Mueller/Stokes-consistent analyzer, conductor Fresnel with geometry, bounded latent proxies, *and* downstream validation against hardware — after reading IEVPF in full |
| 2 | RGB→latent polarization inference | Lin 2025; GenPolar (data-driven/Stokes-informed) | Not novel. Ours differs by explicit latent optical state (D,S,n,r,η,k) feeding a differentiable Fresnel/Stokes layer; claim = *interpretable latent route*, to be compared to data-driven baselines |
| 3 | Differentiable virtual analyzer | trivial given Malus/Mueller; used in many pipelines | **Engineering, not a contribution** |
| 4 | PSRF (circular-harmonic analyzer-response field, order selection) | harmonic Malus fits are standard in polarimetry | Novelty is *use as a robust per-pixel feature and its order selection by held-out angles*; modest |
| 5 | Copper/bronze geometry-aware optical prior | cylinder priors/SfP geometry exist; metal pBRDF measured (Baek 2020) | Possibly new combination for cartridge corrosion; must be shown by H4 |
| 6 | Physics-aware feature fusion (RGB/Lab/HSV + PSRF) | polarization cues as inputs (Kalra 2020) | Incremental; value = controlled colour-space/fusion comparison |
| 7 | Hardware-teacher → RGB-student polarization distillation | no direct match found (search was partial) | **Most promising**, conditional on a real hardware dataset and the gap experiment (H3) |
| 8 | Counterfactual Polarimetric Ensemble (CPE) | ensembles/MC dropout generic; physical-latent ensembles with corrosion-stability test not found | Candidate; must show CPE variance correlates with error (RQ6) and stabilises decisions |
| 9 | Uncertainty-aware corrosion inference | generic | Incremental unless tied to abstention benefits on real data |
| 10 | Physical-vs-virtual benchmark | not found | **Valuable regardless of sign of result** |

**Defensible headline.** Not "software replaces a polarizer" and not "first RGB polarization", but: *a hardware-calibrated, physics-constrained
virtual polarimetric representation for copper/bronze corrosion, and a measurement of how much physical-polarization benefit it recovers.*

**Before any novelty statement in a paper:** (i) read IEVPF, Lin et al., GenPolar in full; (ii) repeat the search in IEEE Xplore, ScienceDirect,
OSA/Optica, SPIE; (iii) complete `NOVELTY_MATRIX.csv` for all `partial_search` rows.
