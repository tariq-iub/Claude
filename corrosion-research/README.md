# Corrosion Recognition — 8 Novel Architectures

Eight architecturally distinct, from-scratch designs for pixel-level
corrosion segmentation/classification/severity estimation on metallic
(copper-alloy) surfaces, each with full theory + math ([`RESEARCH.md`](RESEARCH.md)
and each architecture's own `README.md`) and complete, runnable PyTorch
code in its own folder.

| # | Architecture | Folder | Params | Core idea |
|---|---|---|---|---|
| 1 | PMD-Net | [`architectures/01_pmd_net/`](architectures/01_pmd_net/) | ~7K | Learned-bandwidth anisotropic diffusion PDE |
| 2 | CPEN | [`architectures/02_cpen/`](architectures/02_cpen/) | ~13K | Ordered, EMA-evolving colour prototypes |
| 3 | HyperCorNet | [`architectures/03_hypercornet/`](architectures/03_hypercornet/) | ~4K | Two-level differentiable hypergraph |
| 4 | CorroNCA | [`architectures/04_corronca/`](architectures/04_corronca/) | ~3K | ΔE00-perceiving neural cellular automaton |
| 5 | Corro-SSM | [`architectures/05_corro_ssm/`](architectures/05_corro_ssm/) | ~4K | ΔE00-gated linear state-space model |
| 6 | EB-TopoSeg | [`architectures/06_eb_toposeg/`](architectures/06_eb_toposeg/) | ~7K | Unrolled energy minimisation + topology penalty |
| 7 | INCF | [`architectures/07_incf/`](architectures/07_incf/) | ~18K | Colour-conditioned implicit neural field |
| 8 | MoMER | [`architectures/08_momer/`](architectures/08_momer/) | ~17K | Colour-prior-routed mixture of experts |

All 8 fit comfortably on an Intel i7 / 24GB RAM / ~2GB VRAM machine (all
under 20K parameters; see `RESEARCH.md §10`).

## Start here

- **Theory, math, novelty audit, comparison matrix, literature-search plan,
  benchmarking/statistical protocol, publication framing** → [`RESEARCH.md`](RESEARCH.md)
- **Per-architecture deep dive** (hypothesis, math, blocks, losses,
  ablations, expected strengths/weaknesses) → each `architectures/<name>/README.md`

## Quickstart

```bash
pip install -r requirements.txt

# smoke-test a single architecture's forward pass + parameter count
python architectures/01_pmd_net/model.py

# train one architecture on the bundled synthetic dataset
python architectures/01_pmd_net/train.py --epochs 15 --cpu

# run inference with a trained checkpoint (or random weights, for a smoke test)
python architectures/01_pmd_net/infer.py --checkpoint architectures/01_pmd_net/runs/seed0/model.pt --cpu

# run the full benchmark: all 8 architectures + Tiny-U-Net + classical baselines
python benchmarks/run_benchmark.py --epochs 10 --seeds 0 1 2 --cpu
```

Every architecture folder follows the same layout:

```
architectures/<NN>_<name>/
├── model.py    # nn.Module + build_model() factory; `python model.py` self-tests shape/params
├── train.py    # thin wrapper around common/engine.py::train
├── infer.py    # loads a checkpoint, runs inference, saves a colourised prediction
└── README.md   # full 25-point architecture specification
```

## No real dataset is bundled

There is no public, pixel-annotated, license-clear copper-corrosion dataset
included. `common/dataset.py::SyntheticCorrosionDataset` procedurally
generates plausible training images (metal base + irregular oxide blobs
with class-consistent Lab statistics, illumination/shadow/dirt
augmentation) so every architecture and the whole benchmark pipeline is
runnable and testable end-to-end today. Swap in real data via
`common/dataset.py::FolderCorrosionDataset` (`images/*.png` +
`masks/*.png`) once an annotated corpus is available — see
`RESEARCH.md §7` for the recommended split/labelling protocol.

## Caveat on results

Numbers this code produces on the synthetic dataset validate that every
architecture is **differentiable, trainable and correctly wired**, not that
it beats anything on real corrosion imagery. Every performance/novelty
claim in `RESEARCH.md` is explicitly framed as a hypothesis pending the
literature/patent search and the real-data experimental protocol described
there — see `RESEARCH.md §6, §8`.

## Polarization inputs (experimental, synthetic-only so far)

Optional measured-polarization channels (DoLP, normalised Stokes, AoLP unit vector, raw analyzer stack) built on
`virtual_polarimetric_corrosion` (`vpc`) physics; see the design doc in the project files (`polarization-extension/DESIGN.md`).
Default behaviour is unchanged (`FolderCorrosionDataset(pol_mode="none")`).

```bash
python tools/export_vpc_synthetic.py --out data/vpc_synth --groups 24 --views 3   # vpc render -> 6-class folder dataset + pol/*.npz
python benchmarks/run_pol_wiring.py --data data/vpc_synth --epochs 8 --seeds 0 1 --cpu
python -m pytest -q tests
```

Caveats: no polarization hardware or real polarization data exists yet; the synthetic generator itself puts class signal in the
polarization channels, and the 7-to-6 class mapping in `tools/export_vpc_synthetic.py` is an assumption. The wiring benchmark
checks plumbing only. Its `*_shuffled` control arms (polarization taken from a different sample) matched or beat the real arms in
a small run, so those numbers show no polarization benefit; the fusion head itself adds capacity.
