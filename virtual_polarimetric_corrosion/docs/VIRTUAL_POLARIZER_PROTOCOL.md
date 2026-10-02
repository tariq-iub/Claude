# Virtual polarizer protocol

**Definition.** Given a single sRGB image (and optionally silhouette-derived cylinder normals), produce *estimated* polarization-conditioned observations: Ŝ0, Ŝ1, Ŝ2 (per colour channel), analyzer images Î_θ for any θ ∈ [0°,180°), DoLP_hat, AoLP_hat, PSRF coefficients, plus an ensemble (CPE) describing plausible alternatives. Output ≠ measurement.

## Implementations (increasing capability)
1. `pseudo_polarizer` — strawman brightness filter (no physics); *must* lose to everything polarization-aware or the physics is not helping.
2. `physics_only_virtual_polarizer` — specular-free decomposition with metal-tinted specular colour, roughness proxy, cylinder normals, Fresnel Stokes, analyzer (no learning).
3. `VPCorrosionNet` optical branch — learned latent Z + `PhysicsStokesLayer` (environment/directional), optional learned residual polarization.
4. Distilled student — (3) trained with measured Stokes targets from the hardware teacher.

## Validation levels (never skip a level)
| Level | Test | Data | Pass criterion |
|---|---|---|---|
| L0 | closed-form identities: Malus, periodicity, I_θ+I_{θ+90}=S0, 4-angle Stokes, PSRF order 1/2, Jones→Mueller | analytic | tests pass (float precision) |
| L1 | physics modules vs renderer truth (`run_synthetic_validation.py`) | SYNTHETIC | identities ≤ 1e-4; baselines reported |
| L2 | virtual vs **measured** analyzer images/DoLP/AoLP on dielectric & polished-metal references | REAL (reference targets) | within calibrated uncertainty or reported gap |
| L3 | virtual vs measured on corroded cartridges (held-out groups) | REAL | beats trivial baselines (gate G3); circular-safe AoLP |
| L4 | downstream: corrosion segmentation/severity under glare | REAL | H1 |

## Procedure
1. Linearize; mask clipped pixels (g). 2. Fit silhouette → cylinder prior (optional). 3. Run optical branch → Z; physics layer → Ŝ. 4. `VirtualAnalyzer` on a θ grid (8 canonical or dense 5°). 5. PSRF fit; select order by hold-out angles when ≥ 8 measured angles exist.
6. `optimise_analyzer` → θ* with J(θ) (terms min–max normalised; periodic PCHIP; samples shown as markers, fit as a separate curve; weights chosen on validation data).
7. CPE with K ≥ 16 → mean/variance/credible interval; report AoLP circular variance. 8. Abstain where uncertainty exceeds the validation-chosen threshold.

## Reporting rules
Always label outputs "estimated"; plot measured vs estimated with markers/colour distinct; AoLP with cyclic maps and wrapped error; show DoLP where AoLP is meaningful (DoLP > threshold); never show virtual cross-polarized results as validated until the frame-convention issue (POLARIZATION_PHYSICS §7) is closed.
