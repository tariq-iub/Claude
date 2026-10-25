# cucorr: automated copper-corrosion annotation and explainable segmentation

This project segments copper-surface images into six visual classes: **Healthy, Cu2O, CuO, CuCl, CuCl2, Patina**. It also keeps an explicit
**Unknown/Uncertain** state, supports expert-in-the-loop review and blinded multi-annotator agreement, and benchmarks every method under one statistical protocol.

> No real-data results have been produced yet. The project has no reference masks. The real images (about 500) are not in the repository either.
> All numbers in `outputs/synthetic_example/` come from synthetic fixtures and only check that the pipeline runs.

## Install (CPU baseline)
```bash
cd copper_corrosion
pip install -e ".[dev]"            # optional extras: .[deep] .[sam] .[app] .[xai]
pytest -q
```

## Commands
```bash
cucorr audit                                      # workbook audit -> docs/audit/
cucorr predict --approach approach_a --images data/images        # no training needed
cucorr predict --approach approach_b --images data/images \
       --train-images data/annotations/train_images --train-masks data/annotations/adjudicated
cucorr benchmark --images data/images --masks data/annotations/adjudicated \
       --splits data/annotations/splits.csv --out outputs/bench \
       --approaches A_lab_de2000,B_rf,C_slic_400,F_hybrid --experiments configs/experiments.yaml \
       --reference-type adjudicated_expert
cucorr iaa data/annotations/raters/alice data/annotations/raters/bob --out outputs/iaa.json
cucorr active-learning --images ... --masks ... --out outputs/al.json
streamlit run app/annotation_app.py -- --images data/images --store data/annotations/store
python scripts/make_synthetic_example.py         # end-to-end demo on synthetic data
```
To smoke-test on real photographs, put them in `data/test_images/`. `tests/test_real_images.py` runs Approach A on them and checks the
output contract (it does not measure accuracy).

## Layout
`src/cucorr/`: `data/` (workbooks, splits, synthetic), `color_science/`, `segmentation/` (A–F behind one interface),
`active_learning/`, `annotation/` (append-only store, edit ops), `evaluation/` (metrics, agreement, statistics, benchmark),
`visualization/`, `reporting/`. Docs live in `docs/`: start with `audit/audit.md`, then `data_dictionary.md`, `methodology.md`,
`annotation_protocol.md`, `evaluation_protocol.md` and `research_paper_outline.md`.

## Status of approaches
| Approach | Status |
|---|---|
| A: colour matching (RGB / CIE76 / CIEDE2000, multi-prototype, abstention) | implemented and tested |
| B: RF / SVM / HistGB pixel classifiers | implemented and tested |
| C: SLIC + region RF + edge-aware graph smoothing | implemented and tested |
| D: U-Net / DeepLabV3+ (SMP) with a tiny U-Net fallback, Dice+CE, MC dropout | implemented. The test needs `torch` (skipped where it is missing) |
| E: SAM 2 adapter + class assigner | adapter implemented. Needs `sam2`, a checkpoint and ideally a GPU. Tested with a classical proposer only |
| F: hybrid ensemble + active learning (random / uncertainty / disagreement) | implemented and tested |
