# Reproducibility

- **Configuration:** `configs/default.yaml` holds every setting, and `experiments.yaml` lists the deltas. The SHA-256 of the merged config is
  written to every `stats.json` and `annotation_metadata.json`.
- **Seeds:** the global `seed` drives sampling, splits, model initialisation and bootstrapping.
- **Provenance:** each output records the input image SHA-256, the framework version, the approach description (including the backend,
  the supervision type and whether the result is provisional) and a UTC timestamp.
- **Environment:** `requirements-lock.txt` lists the CPU-baseline versions that were verified. Extras are `[deep]`, `[sam]`, `[app]` and `[xai]`.
- **Private data:** `data/images/`, `data/annotations/` and `data/private/` are git-ignored, along with `outputs/*` (the synthetic
  example is the one exception) and model checkpoints. Keep expert annotations in access-controlled storage, and share them through DOI-backed
  archives only with consent.
- **Checkpoints:** record the source URL, the SHA-256 and the licence in `checkpoints/README.md` next to each file.
