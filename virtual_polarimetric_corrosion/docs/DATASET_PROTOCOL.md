# Dataset protocol (copper/bronze cartridges)

**Material safety/legal.** Spent cases or inert/dummy rounds only; follow local regulations for storage and transport; no live ammunition. Record alloy (copper/brass/bronze — cartridge cases are often brass; if unknown, run XRF/spark-OES on a subset) because it sets the optical constants.

## Units and identifiers
`group_id` = physical cartridge (never shared across splits); `session_id` = acquisition session; `view_id`. Near-duplicate frames (burst, same pose) share the group and never cross splits.

## Diversity targets (record as metadata columns)
corrosion level (0–4 placeholder bins) and subtype (discoloration, oxidation-dark, patina, chloride-like deposit, pitting — *visually annotated, not chemically identified*), surface finish (polished/as-fired/handled), cartridge geometry/calibre/length, orientation (axis angle, roll), illumination (type, direction, intensity, colour temperature, distance), camera (model, sensor, lens), exposure/ISO/white balance, background, specular highlight intensity (fraction of saturated pixels), distance.

## Hardware-teacher subset
Every sample in this subset gets, in addition to a normal RGB image, linear analyzer images at 0/45/90/135° (and optionally 22.5…157.5°), with identical camera settings and registration (HARDWARE_ACQUISITION_PROTOCOL.md). ≥ 30 % of cartridges recommended; teacher/student comparisons use only held-out cartridges.

## Annotation
Classes: background, healthy_metal, discoloration, oxidation_dark, patina_green, chloride_deposit, pitting (+ pit mask). Two annotators + adjudication on a 20 % overlap; report Cohen's κ / pixel IoU between annotators; ambiguous boundaries get an "ignore" band. Severity: area-fraction bins fixed by experts (placeholder cut-points 1/5/15/30 %). Chemical ground truth, if any, is stored separately and never inferred from colour.

## Splitting
70/15/15 by group, stratified by cartridge severity bin (`splits.group_stratified_split`, remainder-carrying to hold proportions). Leakage checks: `assert_no_group_leakage`, `near_duplicate_check` (embedding cosine) across groups, session-overlap report. External validation set: different camera/site/lot.

## Manifest schema (`RealManifestDataset`)
`image_path, mask_path, group_id, session_id[, I0_path, I45_path, I90_path, I135_path, I22_5_path, ..., severity_level, severity_fraction, split]`. Table `dataset_summary` is filled by script from the manifest.

## Datasheet
Collection dates, devices, calibration files, annotator instructions, known biases (e.g., lighting uniformity, background), licence.
