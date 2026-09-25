# Novel Corrosion-Recognition Architectures — Research Program

This document is the master research artifact for the project in this
repository: **8 architecturally distinct proposals** for pixel-level
corrosion segmentation, classification, boundary localisation and severity
estimation on metallic (copper/copper-alloy) surfaces, each accompanied by
**complete, runnable PyTorch code** in its own folder under
[`architectures/`](architectures/).

> **Status discipline.** Nothing below is an experimentally verified claim
> of superiority. Numbers produced by the code in this repo on the bundled
> *synthetic* dataset (see §7) are smoke-test sanity checks that the
> architectures are trainable and differentiable end-to-end — they are
> **not** evidence about real corrosion imagery. Every "novel," "first," or
> "SOTA-surpassing" characterisation below is explicitly a **hypothesis
> pending the literature/patent search (§6) and the experimental protocol
> (§7–§9)**, per the task's own instructions.

---

## 1. Problem framing and class taxonomy

Six-class per-pixel taxonomy used throughout:

| id | class | typical appearance |
|----|-------|---------------------|
| 0 | Healthy / uncorroded | bright metallic copper |
| 1 | Cu₂O (cuprite) | red-brown, often the earliest oxide |
| 2 | CuO (tenorite) | near-black, dense crust |
| 3 | CuCl (nantokite) | pale grey-green, powdery, chemically unstable ("bronze disease") |
| 4 | Cu₂Cl(OH)₃-type (atacamite family) | bright green, bloom-like |
| 5 | Ignore/uncertain | annotation-ambiguous transition pixels |

Two coordinate systems recur throughout the mathematics below:

- **Spatial distance**: `d_s(i,j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)`
- **Perceptual colour distance**: `d_c(i,j) = ΔE00(Lab_i, Lab_j)` (CIEDE2000)

`common/colorspace.py` implements a fully differentiable CIEDE2000 (not the
crude Euclidean-Lab approximation), so every architecture that "uses ΔE00"
can do so as a genuine part of the computational graph — directly
differentiable, not merely as a frozen pre-processing step. Each
architecture's README states explicitly whether ΔE00 is: **(a) directly
differentiable and computed exactly, (b) approximated by a cheaper learned
surrogate, (c) used for loss supervision, (d) used for gating/routing, or
(e) used for neighbourhood/graph construction** — see the per-architecture
"Colour usage" section.

---

## 2. The 8 proposed architectures — one-paragraph summaries

| # | Name (acronym) | Family | Core computational primitive |
|---|------|--------|-------------------------------|
| 1 | [PMD-Net](architectures/01_pmd_net/) | Perceptual-field network | Anisotropic reaction-diffusion PDE with learned spatial/perceptual bandwidths, integrated via explicit Euler steps (not attention) |
| 2 | [CPEN](architectures/02_cpen/) | Prototype-evolution network | Ordered, EMA-evolving prototype trajectories per class + nearest-prototype read-out |
| 3 | [HyperCorNet](architectures/03_hypercornet/) | Hypergraph corrosion model | Two-level differentiable hypergraph (pixel↔cell, cell↔colour-bin) with spectral hypergraph convolution |
| 4 | [CorroNCA](architectures/04_corronca/) | Cellular-automata segmentation | Locally recurrent, weight-shared update rule with explicit ΔE00-to-neighbour perception channels |
| 5 | [Corro-SSM](architectures/05_corro_ssm/) | State-space corrosion model | Diagonal linear SSM scanned row/column-wise, forget gate = closed-form function of ΔE00 |
| 6 | [EB-TopoSeg](architectures/06_eb_toposeg/) | Energy-minimisation / boundary-topology network | Unrolled gradient descent on a learned energy field + differentiable soft-morphological topology penalty |
| 7 | [INCF](architectures/07_incf/) | Implicit-function segmentation | Coordinate-based SIREN field conditioned on sampled latent + local Lab, continuously queryable at any resolution |
| 8 | [MoMER](architectures/08_momer/) | Mixture-of-material-experts | Expert routing gate = learned logit × explicit per-expert Gaussian Lab-prior likelihood; routing entropy = native uncertainty |

Each architecture's own README (linked above) answers, in full, the
25-point specification required by the task brief (hypothesis, novelty
justification, input representation, feature encoding, blocks, information
flow, math, inductive bias, colour usage, losses, uncertainty, boundary
refinement, multiclass strategy, training/inference procedure, complexity,
parameter count, FLOPs, memory, strengths, weaknesses, ablations, baselines,
statistical protocol).

---

## 3. Internal novelty audit

For every candidate, the design was checked against the disqualifying
patterns named in the task brief (`existing arch + new loss`,
`existing backbone + Lab input`, `U-Net + attention`, `Transformer + ΔE00`,
`CNN + clustering`). The table states the **actual new computational
primitive** claimed, and the closest prior-art family it must be
differentiated from experimentally.

| Architecture | Rejected framing | Actual claimed primitive | Closest prior family to differentiate from |
|---|---|---|---|
| PMD-Net | "bilateral filter as a layer" | *Learned-bandwidth anisotropic diffusion PDE integrated over T Euler steps, jointly trained with a calibration loss toward exact ΔE00* | Non-local means / bilateral filtering; guided-filter layers; graph-attention (must show the closed-form Gaussian kernel + PDE-integration framing differs from softmax content-attention) |
| CPEN | "ProtoNet + Lab" | *Ordered EMA-evolving prototype trajectory with an explicit trajectory-smoothness regulariser, modelling oxidation as a manifold walk* | Few-shot prototype networks; VQ-VAE codebooks (neither imposes ordinal/trajectory structure) |
| HyperCorNet | "SLIC + GNN" | *Differentiable, jointly-trained soft-SLIC generating a live incidence matrix feeding a two-level (local+global) hypergraph convolution* | Hypergraph neural networks (Feng et al.); superpixel-GNN pipelines (two-stage, non-differentiable SLIC) |
| CorroNCA | "U-Net + recurrence" | *Weight-shared, locally-recurrent cellular automaton with ΔE00-to-neighbour perception, used for recognition (fixed-point read-out) rather than pattern generation* | Growing/generative NCA (Mordvintsev et al.); recurrent segmentation nets (ConvLSTM-UNet) |
| Corro-SSM | "Mamba + ΔE00" | *Forget gate is a closed-form deterministic function of ΔE00 (not a learned function of hidden state), diagonal SSM scanned separably 2D* | Mamba/S4/vision-SSM (learned-only gates); must show the physically-tied gate changes behaviour, not just naming |
| EB-TopoSeg | "CRF-as-RNN" | *Explicit unrolled gradient descent on a predicted energy field over the label simplex + differentiable soft-morphological-opening topology term* | CRF-as-RNN / DeepLab-CRF (discrete mean-field message passing, no soft-opening topology term); persistent-homology segmentation losses (heavier, non-differentiable-friendly) |
| INCF | "SIREN decoder" | *Field conditioned jointly on sampled CNN latent AND sampled raw Lab at the exact query coordinate, enabling training-free super-resolution mask query* | Implicit neural representations / local implicit image function (LIIF); must show the Lab-conditioning and segmentation-specific framing differ from photo/shape INRs |
| MoMER | "MoE + colour features" | *Routing probability is the product of a learned gate and an explicit parametric (Gaussian) Lab-likelihood, and routing entropy is used directly as the uncertainty output with no auxiliary head* | Sparse MoE vision models; mixture-density uncertainty (separately, not fused into a single routing quantity) |

**Audit verdict**: none of the eight reduces to a single named
`existing-arch + trivial-addition` pattern; each introduces at least one
computational primitive (an operator, a coupling between colour metric and
network dynamics, or an inference procedure) not present as such in the
excluded families (CNNs, U-Net/DeepLab/HRNet/PSPNet/Fast-SCNN/BiSeNet/
MobileNet-seg, ViT/Swin/SegFormer/Mask2Former, Mamba/SSM-vision, GNN/
hypergraph nets, neural fields, prototype nets, differentiable clustering,
NCA, SAM-style prompting, colour-LUT/voxel methods). This is a design-time
self-audit, not a literature verification — §6 specifies how to verify it.

---

## 4. Scientific-novelty ranking matrix

Scored 1 (low) – 5 (high) by the design team's own judgement; **these are
qualitative planning scores, not measured results**.

| Architecture | Conceptual novelty | Math originality | Domain relevance | Feasibility | Compute efficiency | Interpretability | Small-data suitability | Publication potential | Prior-art overlap risk (lower=better) |
|---|---|---|---|---|---|---|---|---|---|
| PMD-Net | 4 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | 3 |
| CPEN | 4 | 3 | 5 | 5 | 5 | 5 | 5 | 4 | 3 |
| HyperCorNet | 5 | 5 | 4 | 3 | 4 | 3 | 3 | 5 | 3 |
| CorroNCA | 5 | 3 | 4 | 4 | 5 | 3 | 3 | 4 | 2 |
| Corro-SSM | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 3 |
| EB-TopoSeg | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 4 |
| INCF | 3 | 3 | 3 | 4 | 3 | 3 | 3 | 3 | 4 |
| MoMER | 3 | 3 | 4 | 5 | 4 | 5 | 4 | 3 | 3 |

**Reading the "overlap risk" column**: EB-TopoSeg and INCF score higher
overlap risk because CRF-refinement networks and implicit neural
representations are both active, crowded literatures; PMD-Net, CPEN,
HyperCorNet, CorroNCA and MoMER combine their primitives with the
corrosion/colour-science domain coupling in ways less densely covered, but
**this is a prior, not a verified, ranking** — see §6.

---

## 5. Shortlist for full experimental implementation

Based on the matrix in §4 (novelty × feasibility × domain relevance,
weighted toward architectures buildable and testable within the stated
hardware envelope), the shortlist for the deepest experimental and ablation
investment is:

1. **PMD-Net** — best efficiency/novelty/interpretability balance; the
   clearest mathematical story (a PDE with two physically meaningful
   bandwidths) for a methods paper.
2. **CPEN** — strongest fit to *severity estimation* (a required objective
   in the brief) via the ordered-prototype trajectory, and best
   interpretability.
3. **HyperCorNet** — highest conceptual/mathematical novelty score; the
   most defensible "new computational primitive" claim (two-level
   differentiable hypergraph), worth the extra implementation risk.

MoMER (native uncertainty) and CorroNCA (extreme compute efficiency, <0.1M
params) are recommended as **secondary** full-experiment candidates
specifically to cover the brief's uncertainty-modelling and
ultra-lightweight-hardware requirements, respectively, even though they
score slightly lower on the composite novelty axis.

This is a prioritisation, not an exclusion — every architecture in this
repo ships with complete, working training/inference code and is equally
runnable.

---

## 6. Literature and prior-art verification plan

For each shortlisted architecture, before any "novel"/"first" claim is
made in a manuscript, run the following search matrix across Google
Scholar, IEEE Xplore, ScienceDirect, SpringerLink, ACM Digital Library,
arXiv, Google Patents and WIPO Patentscope.

### PMD-Net
- `"anisotropic diffusion" segmentation learned bandwidth CIELAB`
- `learned bilateral kernel segmentation network corrosion OR rust`
- `differentiable diffusion PDE semantic segmentation neural network`
- `patent: adaptive diffusion image segmentation learned sigma color space`

### CPEN
- `prototype evolution ordinal trajectory semantic segmentation`
- `VQ-VAE codebook trajectory smoothness classification material degradation`
- `oxidation stage prototype embedding corrosion classification`
- `patent: prototype-based material state classification neural network`

### HyperCorNet
- `differentiable SLIC superpixel hypergraph convolution segmentation`
- `two-level hypergraph pixel superpixel color similarity segmentation`
- `hypergraph neural network corrosion OR rust OR material degradation`
- `patent: hypergraph based image segmentation superpixel color clustering`

### CorroNCA
- `neural cellular automata semantic segmentation recognition (not generation)`
- `growing neural cellular automata perception color difference CIEDE2000`
- `cellular automaton corrosion detection neural network`
- `patent: cellular automata based image classification material defect`

### Corro-SSM
- `state space model gated color difference image segmentation`
- `Mamba vision perceptual gating forget gate color metric`
- `linear recurrent model CIEDE2000 gating segmentation`
- `patent: state space sequence model image segmentation adaptive gating`

### EB-TopoSeg
- `energy based segmentation unrolled gradient descent topology preserving`
- `differentiable morphological opening topology loss segmentation`
- `CRF as RNN alternative soft morphology topology regularization`
- `patent: energy minimization neural network image segmentation topology`

### INCF (secondary)
- `implicit neural representation semantic segmentation local color conditioning`
- `SIREN segmentation arbitrary resolution mask query`
- `local implicit image function segmentation super-resolution mask`

### MoMER (secondary)
- `mixture of experts routing color prior gaussian likelihood segmentation`
- `mixture of experts uncertainty routing entropy semantic segmentation`
- `patent: mixture of experts material classification color routing`

**Outcome protocol**: if a search surfaces a paper/patent sharing the
*same* computational primitive (not just a similar buzzword), the
overlapping mechanism must be identified explicitly, the architecture
redesigned to introduce a materially distinct primitive, and the novelty
table in §4 updated. This repository's code and documentation should be
treated as a design proposal awaiting that verification step, not as a
verified claim of originality.

---

## 7. Dataset protocol

No public, pixel-annotated, license-clear copper-corrosion dataset ships
with this repo. Two tracks:

1. **Synthetic (bundled, `common/dataset.py::SyntheticCorrosionDataset`)** —
   procedurally rendered images with class-consistent Lab statistics per
   oxide, irregular blob boundaries, an "ignore" ring at oxide transitions,
   illumination gradients, cast shadows and dirt/specular noise. Used for
   (a) verifying every architecture is trainable/differentiable end-to-end,
   and (b) fast iteration on ablations before committing GPU time to real
   data. **Not a substitute for real validation.**
2. **Real data (`common/dataset.py::FolderCorrosionDataset`)** — drop-in
   loader for `images/*.png` + `masks/*.png` (matching stems, integer class
   ids matching §1's taxonomy) once an annotated copper-corrosion corpus is
   assembled (e.g. controlled accelerated-corrosion imaging under varied
   illumination, or curated heritage/industrial inspection photographs with
   expert pixel labels). Recommended protocol once real data exists:
   - stratified 60/20/20 train/val/test split by *specimen*, never by
     image crop (prevents leakage between crops of the same physical
     sample);
   - report class frequencies and design the CE class weights accordingly
     (oxide classes are typically much rarer than "healthy");
   - a held-out "cross-illumination" test partition specifically varying
     light colour temperature and shadow severity, to probe the robustness
     objective in the brief.

---

## 8. Benchmarking protocol

### 8.1 Comparison set
Deep baselines: Fast-SCNN, BiSeNetV2, DeepLabV3+/MobileNet, a lightweight
SegFormer variant, U-Net, Attention U-Net, a lightweight Transformer
baseline (bundled: `TinyUNet` in `benchmarks/baselines.py` as the concrete
runnable U-Net reference; the others are external repos to be wired into
the same `common.engine.train`/`evaluate` interface — same input tensor
convention, so integration is a `build_model()` shim).
Classical baselines (bundled and runnable): GMM, K-Means, DBSCAN, SLIC +
Lab-centroid classification, classical CIELAB+ΔE00 nearest-centroid
classification, 3D-RGB-LUT nearest-centroid, 5D-Lab-xy voxel
nearest-centroid (`benchmarks/baselines.py`).

### 8.2 Metrics (per `common/metrics.py`)
Per-class IoU, mean IoU, Dice, Precision, Recall, F1, Accuracy, Boundary
IoU, Boundary F1, Hausdorff distance, Expected Calibration Error (ECE),
inference latency/FPS (CPU and GPU separately), peak VRAM, parameter count.
FLOPs should be measured with `thop`/`fvcore` (not bundled, to keep the
dependency list minimal) using the same input resolution across all models.

### 8.3 Protocol
- **5 fixed random seeds** {0,1,2,3,4}; report **mean ± standard
  deviation** for every metric (`common/engine.py::train` and
  `benchmarks/run_benchmark.py` already parametrise `seed`).
- Same train/val/test split across all models and seeds (seed only
  perturbs initialisation/data order, not the split, once real data is
  used).
- CPU-inference numbers measured with `torch.set_num_threads(1)` for a
  fair single-core comparison in addition to default multi-threaded
  numbers.

### 8.4 Statistical testing
Paired Wilcoxon signed-rank test on **per-image mIoU** between the top
candidate and every baseline, Bonferroni-corrected significance threshold
`α = 0.05 / n_baselines`. Report effect size (matched-pairs rank-biserial
correlation) alongside the p-value, not the p-value alone.

---

## 9. Required ablation studies (strongest candidate)

Applies the brief's mandated ablation list to whichever of the shortlisted
three architectures is selected as the paper's primary contribution
(recommended default: **PMD-Net**, for its clean 2-parameter kernel story):

- Colour input: RGB-only / Lab-only / HSV-only / Lab+HSV / Lab+HSV+chroma+intensity
- With/without spatial coordinates
- With/without the (exact or surrogate) ΔE00 component
- With/without prototype memory *(cross-architecture ablation: swap in CPEN's prototype module)*
- With/without a topology module *(cross-architecture ablation: swap in EB-TopoSeg's soft-opening term)*
- With/without boundary refinement (the shared `boundary_loss` term)
- With/without uncertainty routing *(cross-architecture ablation: swap in MoMER's router)*
- With/without illumination augmentation (`SyntheticCorrosionDataset(illumination_aug=...)`)
- Alternative neighbourhood definitions (window radius `r`, or grid size for HyperCorNet)
- Alternative loss combinations (CE-only, CE+Dice, full `CombinedSegLoss`, + ΔE00-calibration term)

Each architecture's own README lists the subset of this list most relevant
to its own mechanism, plus architecture-specific ablations (e.g. unroll
depth `K` for EB-TopoSeg, expert count `E` for MoMER, scan direction for
Corro-SSM, automaton steps `T` for CorroNCA, query resolution for INCF).

---

## 10. Hardware-constrained deployment notes

All 8 architectures were designed and measured to be trainable on an
Intel-i7-class CPU / 24GB RAM / ~2GB-VRAM GPU:

| Architecture | Params | Tier |
|---|---|---|
| CorroNCA | ~3.2K | <0.1M |
| Corro-SSM | ~3.8K | <0.1M |
| HyperCorNet | ~4.4K | <0.1M |
| PMD-Net | ~7.2K | <0.1M |
| EB-TopoSeg | ~7.4K | <0.1M |
| CPEN | ~13.1K | <0.1M |
| MoMER | ~17.0K | <0.1M |
| INCF | ~17.9K | <0.1M |

(measured via `python architectures/<dir>/model.py`; the bundled `TinyUNet`
baseline is ~0.48M params for scale comparison.) `common/engine.py::train`
supports `amp` mixed precision on CUDA, gradient accumulation
(`grad_accum_steps`), and a `prefer_cpu` flag for CPU fallback; all models
use only `Conv2d`/`Linear`/`GroupNorm`/elementwise ops — no operator
requires more than a few hundred MB of activation memory at 128×128 and
batch size 8, comfortably inside a 2GB VRAM budget. Activation
checkpointing (`torch.utils.checkpoint`) is a straightforward addition to
`common/engine.py::train` for anyone pushing to larger resolutions/batches
than tested here.

---

## 11. Publication framing

### Candidate contribution statements
- "We introduce a family of eight computationally distinct corrosion
  segmentation architectures built from novel primitives — anisotropic
  learned-bandwidth diffusion, ordered prototype evolution, two-level
  hypergraph convolution, ΔE00-perceiving cellular automata,
  perceptually-gated state-space recurrence, unrolled energy minimisation
  with topology regularisation, colour-conditioned implicit fields, and
  colour-prior-routed mixtures of experts — and a shared, reproducible
  benchmarking harness for evaluating them against classical and deep
  segmentation baselines."
- "We show [for the selected shortlisted architecture] that tying a core
  architectural mechanism (diffusion bandwidth / prototype trajectory /
  hypergraph incidence / automaton perception / SSM gate) explicitly to the
  CIEDE2000 perceptual colour metric yields [property to be measured:
  parameter efficiency, small-dataset generalisation, calibration] relative
  to colour-agnostic counterparts of equal capacity."

### Candidate paper titles
1. "Perceptual Manifold Diffusion for Lightweight Corrosion Segmentation"
2. "Oxidation as a Trajectory: Prototype Evolution Networks for Corrosion Severity Estimation"
3. "Two-Level Hypergraphs for Long-Range Corrosion Region Recognition"
4. "Learning to Perceive Material Boundaries: Cellular Automata with Perceptual Colour Gradients"
5. "Colour-Gated State-Space Models for Linear-Complexity Corrosion Segmentation"

### Abstract outline
1. Motivation: corrosion inspection needs robust, efficient, interpretable
   pixel-level recognition under uncontrolled illumination and limited
   hardware. 2. Gap: existing CNN/Transformer/graph segmentation models
   treat colour as an input channel, not as a first-class inductive bias.
   3. Method: the proposed primitive and how it couples perceptual colour
   science to the architecture's computation. 4. Setup: synthetic + real
   corrosion benchmark, classical and deep baselines, 5-seed statistical
   protocol. 5. Findings (to be filled in after experiments — do not
   pre-write numbers). 6. Contribution: a new, ablatable, reproducible
   computational primitive for material-state recognition.

### Proposed paper section structure
1. Introduction 2. Related Work (segmentation architectures; colour-science-
   aware vision; corrosion/material-degradation recognition) 3. Method
   (primitive definition, full math, block diagram, pseudocode) 4.
   Corrosion-Specific Inductive Biases 5. Experimental Setup (dataset,
   baselines, metrics, statistical protocol) 6. Results 7. Ablations
   8. Discussion of Failure Cases 9. Limitations 10. Conclusion.

### Reproducibility checklist
- [ ] Fixed seeds {0,1,2,3,4} used for all reported numbers, mean±std reported.
- [ ] Exact package versions pinned (`requirements.txt`); Python/PyTorch/CUDA versions logged.
- [ ] Train/val/test split files (or generation seed) released.
- [ ] Full hyperparameters logged in `runs/<arch>/<seed>/history.json`.
- [ ] Hardware used for timing numbers stated explicitly (CPU model, GPU model, VRAM).
- [ ] Code for every architecture, baseline, loss, and metric is in this
      repository and independently runnable (`train.py` / `infer.py` per
      architecture; `benchmarks/run_benchmark.py` for the full sweep).
- [ ] Statistical test (Wilcoxon) script and raw per-seed, per-image metric
      dumps released alongside aggregate tables.
- [ ] Literature/patent search queries (§6) and their outcomes documented
      before any "novel"/"first" claim appears in a manuscript.

---

## 12. Repository map

```
corrosion-research/
├── RESEARCH.md                  <- this file
├── README.md                    <- quickstart
├── requirements.txt
├── common/                      <- shared, architecture-agnostic code
│   ├── colorspace.py             (differentiable Lab/HSV/ΔE00)
│   ├── dataset.py                (synthetic + real dataset loaders)
│   ├── losses.py                 (CE+Dice+boundary)
│   ├── metrics.py                (IoU/Dice/F1/Boundary/Hausdorff/ECE)
│   └── engine.py                 (shared train/eval/speed-benchmark loop)
├── architectures/
│   ├── 01_pmd_net/               (model.py, train.py, infer.py, README.md)
│   ├── 02_cpen/                  (...)
│   ├── 03_hypercornet/           (...)
│   ├── 04_corronca/              (...)
│   ├── 05_corro_ssm/             (...)
│   ├── 06_eb_toposeg/            (...)
│   ├── 07_incf/                  (...)
│   └── 08_momer/                 (...)
└── benchmarks/
    ├── baselines.py               (classical + Tiny-U-Net baselines)
    └── run_benchmark.py           (trains/evaluates everything, one report)
```

Every `model.py` is independently runnable (`python model.py`) as a
shape/parameter-count smoke test; every `train.py`/`infer.py` pair is a
thin, consistent wrapper around `common/engine.py` so all 8 architectures
share one training loop, one loss, one metrics suite, and one benchmark
runner — the only thing that differs between folders is the architecture
itself.
