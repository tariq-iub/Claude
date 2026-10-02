# Polarization physics for copper/bronze (established vs approximated)

Legend: **[E]** established physics, **[A]** approximation used here, **[P]** proposed modelling assumption (needs validation).

## 1. What the camera receives [E]
Light reflected from a surface has a polarization state that depends on incidence/viewing geometry, the complex refractive index, roughness
(microfacet distribution), and the sub-surface/overlayer composition. Specular reflection from a smooth interface is *partially* linearly
polarized (unpolarized input): perpendicular to the plane of incidence (s) is reflected more than parallel (p) for dielectrics and, away from
the pseudo-Brewster region, for metals. Light that enters and re-emerges from a dielectric layer (diffuse) is *also* partially polarized,
parallel to the plane of emittance (Atkinson & Hancock). Neither "diffuse unpolarized" nor "specular fully polarized" holds.

## 2. Real hardware chain (for reference)
illumination → [optional source polarizer] → surface → [analyzer] → camera sensor → analyzer images I_θ → Stokes (S0,S1,S2) → DoLP, AoLP.
Four angles (0°, 45°, 90°, 135°) give linear Stokes exactly: S0 = (I0+I45+I90+I135)/2, S1 = I0−I90, S2 = I45−I135. More angles give a least-squares fit
with a residual that diagnoses leakage, drift and sensor nonlinearity.

## 3. Conductors (copper, bronze) [E]
Complex index N = n + ik (e^{−iωt}). Fresnel amplitudes: r_s = (cosθi − N cosθt)/(cosθi + N cosθt), r_p = (N cosθi − cosθt)/(N cosθi + cosθt),
cosθt = √(1 − sin²θi/N²) (complex). R = |r|². Metals show (i) high reflectance, (ii) smaller R_s−R_p than dielectrics (Fig. 3b), (iii) a *phase
difference* between s and p (retardance, Fig. 3c) that couples linear and circular states, (iv) strong wavelength dependence: copper's red
reflectance is far higher than blue (n≈0.2, k≈3.9 in R vs n≈1.1, k≈2.1 in B in the RGB-effective table), so specular DoLP is channel dependent
(R: very low; B: highest). **Implication:** the red channel of copper carries little linear polarization except near grazing angles; polarization-based
contrast on copper is expected to come mainly from the blue/green channels, from the cartridge limb, and from corrosion layers (dielectric-like).
[A] The optical constants in `optics/fresnel.py` are RGB-effective proxies (rendering-literature values), *not* sensor-weighted integrals of measured spectra and
*not* measurements of the inspected alloy. Cartridge cases are often brass (Cu–Zn); `bronze_proxy` is the midpoint of Cu and Au tables — an assumption.

## 4. Corroded surfaces [A]/[P]
| Surface state | Optical picture | Expected polarization behaviour (hypotheses, not facts) |
|---|---|---|
| polished healthy metal | smooth conductor, low roughness, specular-dominant | moderate DoLP at oblique angles, strong glare, AoLP ⟂ plane of incidence |
| tarnish / thin oxide | thin film on conductor; interference not modelled | colour change; DoLP shift possible; thin-film effects ignored [A] |
| Cu2O-like / CuO-like (dark red/brown/black) | absorbing dielectric-like layer | weaker specular, diffuse partially polarized (parallel) |
| patina (green), chloride deposits (pale, powdery) | weakly absorbing dielectrics, porous, rough | specular small and strongly depolarized by roughness/volume scattering; diffuse DoLP follows n_d |
| pitting | local roughness, normal discontinuities, occlusion | local DoLP drop + AoLP scatter; shadow/highlight dipole under oblique light |
Subtype names are *visually annotated classes*; no chemical identification is claimed without chemical ground truth.

## 5. Microfacet reflection [E]/[A]
GGX/Trowbridge–Reitz with Smith masking-shadowing; Fresnel evaluated at the microfacet normal (half vector h). Valid under geometric optics (roughness
correlation length ≫ wavelength), single scattering, isotropy. Weak for powdery corrosion (volume scattering), pits near the shadowing length scale, thin films.
For a *distant directional light* the Fresnel angle θd = ∠(h,v) is independent of the surface normal and AoLP_spec = azimuth(v×l) is constant over the
image; the normal only decides *where* the highlight is. For *extended/environment illumination* a mirror-like facet samples the environment around
the mirror direction, so θi = ∠(n,v) and AoLP_spec = azimuth(n) + π/2 [P: environment mode]. Which regime holds is a property of the capture setup and must be
declared per dataset (`net.illumination`).

## 6. Roughness depolarization [P]
Rough facet ensembles mix planes of incidence and depolarize: ρ_eff = ρ_Fresnel·exp(−(r/r0)²) with r0 ≈ 0.5 as a smooth placeholder. r0 can be learned
(`PhysicsStokesLayer(learn_r0=True)`) or fit to hardware data (HARDWARE protocol §6).

## 7. OPEN ISSUE — source/analyzer frame convention (cross-polarized imaging)
`PhysicsStokesLayer.chain_intensity` evaluates source polarizer → surface Mueller → analyzer with both angles referenced to the camera image
plane and the reflection Jones matrix diag(r_s, −r_p) (zero retardance at normal incidence). Verified: extinction/leakage for s- or p-aligned sources
(`tests/test_polarization.py`). **Not resolved:** for a source at 45° to the plane of incidence at oblique/grazing angles the retardance δ → π maps +45° to −45°;
whether a physical "crossed" pair blocks glare then depends on how the source and analyzer zeros are referenced across the mirror reflection (a mirror
reverses the handedness of the transverse frame). Fig. 6 therefore shows the crossed state as *unvalidated*. Resolve on a bench mirror/dielectric plate in
Phase 2–3 before using any virtual cross-polarized output quantitatively. The analyzer-only path used by VP-CorrosionNet is unaffected (camera frame only).

## 8. What a single RGB image can and cannot give
RGB provides 3 numbers per pixel (after unknown tone-mapping, white balance, clipping). The latent state has ≥ 10 unknowns per pixel (D(3), S(3), n(2), r, η, k, ...).
The inverse problem is under-determined; constraints come from physics layers, geometry priors, smoothness and learned priors. Different latent states
produce the same RGB but different I_θ — hence the CPE. Virtual polarimetry ≠ measurement.
