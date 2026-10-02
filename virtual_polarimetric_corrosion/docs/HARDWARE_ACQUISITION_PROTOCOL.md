# Hardware acquisition protocol (real polarization teacher)

## 1. Rig
* Camera: linear sensor (RAW or 16-bit linear), fixed gain/exposure/white balance, no auto features; ideally monochrome or RGB with raw Bayer access. Alternative: division-of-focal-plane polarization sensor (4 angles in one shot; check its extinction ratio and per-pixel calibration).
* Analyzer: linear polarizer in a motorised/indexed rotation mount in front of the lens, angular repeatability ≤ 0.5°, extinction ratio ≥ 100:1 over the spectral band (record). Quarter-wave plate optional for S3 (not required here).
* Illumination: (i) large softbox / integrating dome (environment-like) and (ii) small directional source at a recorded direction; optional source polarizer on a second mount. Record geometry (distances, angles) and keep it fixed per session.
* Object: cartridge on a rotation stage (axis angle and roll recorded); matte non-reflective background; level and clamp.
* Tripod/rigid mounting: no motion between angles (sub-pixel; verify by phase correlation).

## 2. Calibration (per session)
1. **Zero reference** of the analyzer vs camera x-axis: with a second polarizer, find extinction positions; or use a dielectric plate at Brewster angle (known s polarization).
2. **Extinction ratio and transmission** t_max, t_min of each polarizer (fits `VirtualAnalyzer(t_min=...)`).
3. Dark frames, flat field (uniform diffuser), sensor linearity (neutral density steps), exposure bracket for clipping check.
4. **Frame-convention test** (open issue in POLARIZATION_PHYSICS §7): bench mirror and dielectric plate with source+analyzer at 0/45/90°, oblique incidence; record to resolve the cross-polarized chain convention.
5. Colour: colour chart under each illuminant (for Lab/ΔE00 features); white balance fixed.

## 3. Capture sequence per object/pose
RGB without analyzer (same exposure compensation policy), then analyzer at 0, 45, 90, 135 (and 22.5, 67.5, 112.5, 157.5 for the dense subset); repeat the 0° frame at the end (drift check, pass if ≤ 1 % mean change). Keep exposure identical across angles (analyzer passes ≈ half of S0; avoid clipping at the brightest angle; bracket if necessary and record gains).

## 4. Processing (`scripts/process_hardware_stack.py`, `experiments/hardware_io.py`)
Dark/flat correction → registration check → least-squares Stokes (S0,S1,S2) → DoLP, AoLP, fit RMSE per pixel. Quality flags: clipped pixels, low-S0 pixels, fit RMSE above noise → mask for training/evaluation.

## 5. Metadata
Cartridge id, alloy, corrosion annotations, camera/lens/settings, light geometry, polarizer serials and calibration values, temperature/humidity, operator, date.

## 6. Experiments the rig must answer before modelling is trusted
(a) dielectric plate reproduces Fresnel DoLP within the calibrated uncertainty; (b) polished copper/brass DoLP vs angle per channel vs the Fresnel prediction with literature constants (tests optical-constant proxies and r0); (c) cross-polarized convention test; (d) repeatability of DoLP/AoLP across sessions.

## 7. Safety
Spent/inert cases only; eye safety for bright sources; local regulations for handling any ammunition components.
