# Reproducibility

* **Environment:** Python ≥ 3.10; `pip install -r requirements.txt` (torch CPU suffices). Development was done with torch 2.14 (CPU), numpy/scipy/matplotlib/scikit-image/pandas current at the time; pin exact versions in your lock file when you run experiments (`pip freeze > results/env.txt`).
* **Determinism:** `train.set_seed` seeds Python/NumPy/Torch and requests deterministic algorithms (warn-only). Synthetic data are deterministic functions of (seed, group, view). CPU runs are bit-reproducible on the same machine; cross-hardware results may differ at ~1e-6.
* **Config + provenance:** every run writes JSON with config and git hash (`train.save_run`); tables include `data_origin` (SYNTHETIC | REAL); results CSVs are append-only raw rows (per seed), summaries are generated.
* **Tests:** `python -m pytest -q` (≥ 115 tests): optics closed forms, Jones→Mueller, Malus/periodicity/orthogonality, PSRF identifiability, cylinder geometry, renderer physical realizability, losses (circular), metrics/statistics/splits, models (all fusion modes), figures (smoke), tables.
* **Regenerate:** `scripts/run_synthetic_validation.py` → `results/synthetic_validation.csv`; `scripts/make_figures.py`; `scripts/make_tables.py`; training/ablation/robustness/hardware-gap scripts read `configs/*.yaml`.
* **Real data:** manifest CSV + hardware calibration files archived with checksums; never edit raw images; processed stacks via `scripts/process_hardware_stack.py`.
* **Bibliography:** run `scripts/verify_bibliography.py` with network access, review `docs/BIBLIOGRAPHY_CROSSREF.csv`.
* **What is not reproducible from this repo alone:** anything requiring the (not yet collected) cartridge dataset or the hardware rig.
