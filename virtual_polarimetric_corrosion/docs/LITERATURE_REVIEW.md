# Literature review (working document)

## 1. Search protocol and honest coverage statement
Searches were run through a web search tool in this session (not IEEE Xplore/ScienceDirect/OSA directly), with the terms listed in the
task brief (polarization imaging defect detection; polarization metallic surface inspection; shape from polarization; deep SfP;
reflection separation; computational polarimetry; virtual polarizer; polarimetric neural network; single RGB polarization estimation;
inverse rendering polarization; copper/bronze corrosion machine vision; corrosion segmentation; pitting; specular metal inspection).
**This is a seeded, partial review, not a systematic (PRISMA-grade) one.** Publisher pages and arXiv were blocked by the sandbox egress
proxy, so most entries are verified at *title/author/venue/year* level from search snippets only; the primary paper's full text could not be
read. `BIBLIOGRAPHY.csv` has a `verification_status` per entry (`verified_search`, `partial_search`, `from_memory`); `scripts/verify_bibliography.py`
performs the Crossref check and must be run before submission. **No citation below should be copied into a paper without that check.**

## 2. The primary reference (Yu, Wang, Wu 2025, Photonics 12(4):368)
What can be said with confidence (abstract/snippet level): authors Zeyu Yu, Dongyun Wang, Hanyang Wu (Zhejiang Normal University); it
analyses light propagation on complex surfaces and builds a polarization imaging system to suppress glare (improves uniformity, lowers noise);
the polarization unit uses horizontal and vertical polarizers to capture in a single exposure; the detector is MF-YOLOv11 (YOLOv11 +
multi-scale edge information selection MSIS + Focal Modulation); on a self-built set P = 86.1 %, R = 71.1 %, mAP50 = 72.7 %
(+3.9, +2.8, +1.6 points over YOLOv11n). **Not confirmed here** (full text unavailable): light source and polarizer/analyzer models, exact
Stokes/DoLP/AoLP processing, dataset size and defect classes, limitations. Read these before Phase 1 closes. Generic limitations
that follow from the design and are therefore safe to state: needs polarization hardware at inference; detection only (boxes);
single-material focus; no explicit optical model of the surface in the learning stage; no uncertainty.

## 3. Taxonomy
**A. Classical polarization optics.** Jones/Stokes/Mueller calculus and Fresnel theory (Born & Wolf; Chipman, Lam & Young; Goldstein) —
`from_memory`. Conductors need complex index N = n + ik; copper constants from Johnson & Christy (1972) — `from_memory`.

**B. Polarization-based inspection.** Yu et al. 2025 (verified at snippet level). Others surfaced by search (title/PMC id only): micro-scale
porosity detection in reflective metal parts with polarization imaging + deep learning (PMC12156235); IEVPF (below); a Applied Optics
paper on specular-highlight suppression with full-polarization imaging via quarter-wave-plate rotation (not captured with authors).
Common pattern: real polarization hardware at inference + YOLO-type detector; metal but not corrosion.

**C. Shape from polarization.** Kadambi et al. ICCV 2015 (Polarized 3D); Ba et al. ECCV 2020 (Deep SfP); Lei et al. CVPR 2022 (SfP in
the wild); Atkinson & Hancock 2006 (diffuse polarization → orientation); Smith et al. (height-from-polarisation, `from_memory`).
Lesson: azimuth has a π ambiguity, diffuse/specular ambiguity is a π/2 flip, zenith mapping is ambiguous — exactly the ambiguities our
latent model must expose (CPE). All require real polarization input.

**D. Physics-informed polarimetric deep learning / inverse rendering.** Deep SfP; Kalra et al. 2020 (polarization cues improve segmentation —
transparent objects); PANDORA (Dave et al. 2022); NeISF (Li et al. 2024) and NeISF++ (conductors and dielectrics); Zhao et al. 2020 (Polarimetric
MVIR); Baek et al. 2018/2020 and Hwang et al. 2022 (polarimetric SVBRDF/pBRDF; include metals). These need real polarization images,
usually multi-view, and are offline.

**E. Reflection separation.** Shafer 1985 (dichromatic, `from_memory`); Wolff & Boult 1991 (`from_memory`); Nayar, Fang & Boult 1997;
Kajiyama et al. WACV 2023 (partially polarized diffuse *and* specular components). Dichromatic assumes illuminant-coloured specular: true for
dielectrics, false for metals — our module takes a metal-tinted specular colour.

**F. Metallic-surface inspection (non-polarized).** Highlight/glare handling by multi-exposure, structured light, dark-field; not reviewed
in depth here (gap in coverage).

**G. Corrosion image analysis.** RustSEG (2022), Segmenting localized corrosion (J. Electron. Imaging 2019), pitting in gas pipelines (CNN,
98.44 % accuracy reported in the snippet; dataset of 576 000 images), MCD-Net (2024), a benchmark of four segmentation networks for
pixel-level corrosion identification. All appearance-only RGB.

**H. Copper/bronze corrosion assessment.** The closest precedent found: deep-learning semantic segmentation of corrosion compounds on
*microscopic* images of iron and copper heritage items (Oltenia Museum), trained from expert delineations. No polarization; microscopy not field
imaging. Cartridge-case corrosion specifically: nothing found in this search (coverage gap, not evidence of absence).

**I. Edge deployment.** MobileNetV2 (Sandler et al. 2018), ECA-Net (Wang et al. 2020) — `from_memory`; ONNX Runtime benchmarking is
engineering practice.

**J. Uncertainty-aware vision.** MC dropout (Gal & Ghahramani 2016), deep ensembles (Lakshminarayanan et al. 2017), heteroscedastic loss
(Kendall & Gal 2017), evidential learning (Sensoy et al. 2018), calibration (Guo et al. 2017) — `from_memory`.

## 4. RGB→polarization and virtual polarizers (the critical neighbours)
* **Lin, Yuan, Chen — RGB-to-Polarization Estimation: A New Task and Benchmark Study (NeurIPS 2025 D&B).** Defines estimating polarization from a
  single RGB image and benchmarks restoration-style networks (snippets mention Restormer, Uformer).
* **GenPolar — Stokes-Informed Diffusion for Robust Linear Polarization Estimation.** Predicts channel-wise linear Stokes from an RGB intensity
  proxy and derives DoLP/AoLP analytically (snippet-level).
* **IEVPF — virtual polarization filtering + YOLO V5-W for additive-manufacturing defects.** "Virtual manipulation of light polarization" to
  improve defect visibility; internals unread.
Consequence: *"estimating polarization from RGB" and "virtual polarization filtering" are not novel as such.* See `NOVELTY_ANALYSIS.md`.

## 5. Comparison table
| Work | Input | Real polarization? | Physics | DL | Output | Limitation | Difference from proposed |
|---|---|---|---|---|---|---|---|
| Yu et al. 2025 | polarization images | yes (hardware at inference) | light-propagation analysis | improved YOLOv11 | defect boxes | hardware needed; detection only | RGB-only deployment, virtual analyzer, corrosion tasks, uncertainty |
| IEVPF (2025) | ? | ? | "virtual polarization" (unread) | improved YOLOv5 | boxes | unread | to be settled after reading |
| Lin et al. 2025 | RGB | no (training data yes) | little | image-to-image nets | polarization maps | generic scenes; no metals/corrosion | conductor Fresnel, PSRF, CPE, downstream corrosion utility |
| GenPolar | RGB intensity proxy | no | Stokes-informed | diffusion | Stokes/DoLP/AoLP | heavy | edge target; physics layer; corrosion |
| Deep SfP (Ba 2020) | 4-angle polarization | yes | SfP priors | CNN | normals | shape only | different goal |
| SfP in the wild (Lei 2022) | polarization | yes | priors | attention | normals | needs hardware | motivates ambiguity handling |
| Kalra 2020 | polarization | yes | cues as inputs | Mask R-CNN | masks | transparent objects | polarization→segmentation precedent |
| PANDORA / NeISF(++) | multi-view polarization | yes | neural Stokes/radiance fields | neural fields | geometry/material | offline, multi-view | single-frame, edge |
| Baek 2018/2020; Hwang 2022 | Mueller/pBRDF captures | yes | full pBRDF | fitting | pSVBRDF | acquisition-heavy | source of metal pBRDF validation data |
| Nayar 1997; Kajiyama 2023 | polarization images | yes | separation | none/limited | diffuse/specular | hardware; assumptions | we separate without hardware, both parts partially polarized |
| Corrosion CV (2019–2024) | RGB | no | none | CNN/ViT | masks/classes | no glare/optics model | physics-conditioned features |
| Heritage copper corrosion (2023) | microscopy | no | none | segmentation | compound regions | microscopy | field RGB + polarization |

## 6. Gaps identified (details in RESEARCH_GAP.md)
1. No work found that links a *conductor-specific* virtual polarimetric model to *corrosion assessment* of copper/bronze.
2. No work found that quantifies **how much of the physical-polarization benefit for inspection** survives RGB-only deployment.
3. Uncertainty in single-image polarization estimation is rarely tied to downstream inspection decisions.
4. Cartridge-case corrosion datasets/benchmarks not found.
