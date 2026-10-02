# Theoretical foundation

1. **Image formation.** I_c(x) = ∫ E(λ) R(x,λ) Q_c(λ) dλ·G(x) + N, c ∈ {R,G,B}; with R split into diffuse, specular, scatter: I = I_d + I_s + I_sc. We work in
   linear radiance (sRGB inverse) with clipping/white-balance unknown; camera response Q_c is approximated by three representative wavelengths or RGB-effective optical
   constants (explicit approximation; whether channel-dependent Fresnel helps is an experiment, `net.base_material` / channel-wise η,k).
2. **Polarimetric forward model.** S_out = Σ_components M_component S_in with incoherent addition (valid for mutually incoherent beams). Conductors: complex Fresnel (n+ik);
   dielectric layers: Fresnel + diffuse emission model; roughness: GGX + depolarization [P].
3. **Why virtual polarimetry is ill-posed and what makes it useful anyway.** The map latent state → RGB is many-to-one. Useful because (a) many latent
   states share *similar corrosion-relevant features* (stability under CPE is testable), (b) priors from geometry (cylinder) fix the normal field, (c) hardware-supervised
   training (teacher) provides measured targets that constrain the otherwise free latent proxies.
4. **PSRF.** The analyzer-response field R_p(θ) = a0 + a1 cos2θ + b1 sin2θ + (a2 cos4θ + b2 sin4θ) summarises a pixel's response; harmonic 1 equals linear Stokes; harmonic 2 is absent
   from ideal linear-analyzer physics and appears with finite extinction ratio or sensor effects — kept only if held-out-angle error supports it.
5. **Learning principle.** Physics layers are differentiable (Fresnel, Stokes, Mueller, analyzer, PSRF), so physics constraints (realizability |(S1,S2)| ≤ S0, unit normals, energy
   bound, periodicity, reconstruction D+S ≈ I) shape the latent network rather than being post-hoc filters. Distillation transfers measured Stokes/feature targets from a teacher.
6. **Evaluation principle.** Physical agreement (circular-safe AoLP error), downstream utility (segmentation/severity under glare), calibration (ECE, risk–coverage), and cost
   (CPU latency) are all reported; accuracy alone is not the objective.
