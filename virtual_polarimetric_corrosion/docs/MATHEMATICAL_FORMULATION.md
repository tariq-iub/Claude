# Mathematical formulation

Tags: **[E]** established physics, **[A]** approximation, **[P]** proposed modelling choice. Conventions: time dependence e^{−iωt}; camera frame x right, y up, z toward the camera;
analyzer/polarization angles measured counter-clockwise from +x; Stokes S0 = I_xx + I_yy, S1 = I_xx − I_yy, S2 = 2Re(E_x E_y*), S3 = −2Im(E_x E_y*) (sign of S3 convention-dependent, unused for linear analysis).
Every code symbol is in `src/vpc/optics/*`, `polarization/*`.

## Part A — Electromagnetic polarization [E]
A monochromatic plane wave travelling along +z: **E**(z,t) = Re{(E_x x̂ + E_y ŷ) e^{i(kz−ωt)}}, E_j = |E_j| e^{iφ_j}. The tip of **E** traces an ellipse set by the amplitude ratio |E_y|/|E_x|
and phase difference δ = φ_y − φ_x. Linear: δ ∈ {0, π}. Circular: |E_x| = |E_y|, δ = ±π/2.

## Part B — Jones calculus [E]
**E** = [E_x, E_y]^T; optical element → 2×2 complex matrix J, **E'** = J**E**. Ideal linear polarizer at θ: J_P(θ) = [[c², cs],[cs, s²]], c = cosθ, s = sinθ (projector). Reflection from a
planar interface in the (s,p) basis: J = diag(r_s, −r_p) (our sign choice, zero retardance at normal incidence). **Limitation:** Jones calculus describes fully polarized *coherent* light; partially polarized
light and incoherent sums require Stokes/Mueller. Here Jones is used for derivation and test (`jones_to_mueller`), not for imaging.

## Part C — Stokes vector [E]
For partially polarized quasi-monochromatic light, with coherency matrix ρ = ⟨**E E**^†⟩: S_i = Tr(σ_i ρ) (σ_0 = I, σ_1 = diag(1,−1), σ_2 = [[0,1],[1,0]], σ_3 = [[0,−i],[i,0]]).
S0² ≥ S1² + S2² + S3² (equality iff fully polarized). DoLP = √(S1²+S2²)/S0 ∈ [0,1]; AoLP = ½ atan2(S2,S1) (period π; S1, S2 → −S1, −S2 corresponds to AoLP + π/2).
Linear-only imaging: S = [S0,S1,S2]^T. Estimated counterparts: Ŝ0, Ŝ1, Ŝ2, DoLP_hat, AoLP_hat.
Realizability constraint used as a loss/clip: √(Ŝ1²+Ŝ2²) ≤ Ŝ0.

## Part D — Mueller matrix [E]
S_out = M S_in, 4×4 real. From a Jones matrix: M_ij = ½ Tr(σ_i J σ_j J^†) (implemented). Cascade: M = M_N ⋯ M_1. Incoherent sum of components: S = Σ_k S_k (diffuse + specular + scatter).
Rotation of the reference frame by α: R(α) = [[1,0,0,0],[0,cos2α,sin2α,0],[0,−sin2α,cos2α,0],[0,0,0,1]]; an element defined in its own frame is expressed in the camera frame as R(−α) M R(α).

## Part E — Linear polarizer [E]
Substituting J_P(θ) in the formula of Part D gives
M_P(θ) = ½ [[1, c2, s2, 0],[c2, c2², c2 s2, 0],[s2, c2 s2, s2², 0],[0,0,0,0]], c2 = cos2θ, s2 = sin2θ.
Idempotent (M_P² = M_P), crossed polarizers annihilate (M_P(θ+π/2)M_P(θ) = 0) — both tested. Finite extinction ratio (diattenuator, t_max, t_min): first row ½[(t_max+t_min), (t_max−t_min)c2, (t_max−t_min)s2, 0];
`linear_polarizer_mueller(θ, t_max, t_min)`.

## Part F — Malus law [E]
Measured intensity = first component of S_out = M_P(θ)S: I_θ = ½(S0 + S1 cos2θ + S2 sin2θ). For fully polarized input at θ0: S = I0[1, cos2θ0, sin2θ0]: I_θ = ½I0(1 + cos2(θ−θ0)) = I0 cos²(θ−θ0).
Consequences (all tested): period π, I_θ + I_{θ+π/2} = S0, S recovered exactly from four angles, harmonic content only 0 and 2θ.
**VirtualAnalyzer(θ)** implements this for Ŝ (differentiable), optionally with extinction ratio.

## Part G — Fresnel equations, dielectrics [E]
n1 sinθi = n2 sinθt. r_s = (n1 cosθi − n2 cosθt)/(n1 cosθi + n2 cosθt), r_p = (n2 cosθi − n1 cosθt)/(n2 cosθi + n1 cosθt) (Born & Wolf sign: r_p = −r_s at normal incidence in our convention).
R_s = |r_s|², R_p = |r_p|². Brewster: θ_B = atan(n2/n1), R_p(θ_B) = 0. Normal incidence R = ((n−1)/(n+1))². Total internal reflection handled by the complex square root. Specular DoLP for unpolarized input:
(R_s − R_p)/(R_s + R_p).

## Part H — Metallic Fresnel reflection [E]
N = n + ik. cosθt = √(1 − sin²θi / N²) (principal branch), r_s = (cosθi − N cosθt)/(cosθi + N cosθt), r_p = (N cosθi − cosθt)/(N cosθi + cosθt), R_{s,p} = |r_{s,p}|²,
retardance δ = arg(−r_p) − arg(r_s) (0 at normal incidence). Copper: n(λ), k(λ) vary strongly across the visible, so R_{s,p} are channel dependent. [A] RGB-effective constants replace ∫ n(λ) Q_c(λ)dλ;
the proper treatment is R_c = ∫ R(λ) E(λ) Q_c(λ) dλ / ∫ E Q_c and requires the camera spectral response; if unknown this approximation is declared and its value tested (channel-wise vs single-index).

## Part I — Microfacet BRDF [E]/[A]
f_s(l,v) = D(h) G(l,v) F(θd) / (4 (n·l)(n·v)), h = (l+v)/‖l+v‖, cosθd = h·v. GGX: D(h) = α²/(π((n·h)²(α²−1)+1)²), α = r². Smith: G = G1(l)G1(v), G1(x) = 2(n·x)/((n·x)+√(α²+(1−α²)(n·x)²)).
Polarized form [E] for the Fresnel term: per-facet Mueller matrix M_F(θd) in the plane-of-incidence frame, rotated to the image frame by ψ = azimuth(v × l):
S_spec = (D G / 4(n·v)) · R(−ψ) M_F R(ψ) S_in. For unpolarized S_in = [1,0,0,0]: S_spec = (DG/4(n·v))·[F̄, ΔF cos2ψ, ΔF sin2ψ], F̄ = (R_s+R_p)/2, ΔF = (R_s−R_p)/2 (cross-checked against the full Mueller path in tests).
[A] valid for single scattering, geometric optics, isotropic roughness; poor for powdery deposits, pits near the masking scale, thin films.

## Part J — Diffuse/specular decomposition [E]/[A]/[P]
Dichromatic model I = I_d + I_s with I_s ∝ illuminant colour — **dielectrics only**; for metals the specular colour is Fresnel-tinted: w_c = F̄(θ, N_c) (default in `physics_only_virtual_polarizer`). Shen–Zheng-type min-channel separation:
s = min_c(I_c/w_c) − floor, I_s = s·w, I_d = I − I_s [A].
Diffuse polarization (Atkinson & Hancock): ρ_d(θ) = (n−1/n)² sin²θ / (2 + 2n² − (n+1/n)² sin²θ + 4cosθ √(n² − sin²θ)), AoLP_d = azimuth(n) (parallel to the plane of emittance) [E for dielectrics].
Environment-illumination specular: θi = ∠(n,v), AoLP_s = azimuth(n) + π/2 [P]; roughness depolarization ρ_eff = ρ_F exp(−(r/r0)²) [P]; scatter fully depolarized [A].
Total: S = S_d + S_s + S_sc, with S_d = D[1, ρ_d cos2a, ρ_d sin2a], S_s = S[1, ρ_eff cos2ψ, ρ_eff sin2ψ] (radiances D, S per channel).

## Part K — Virtual polarization estimation [P]
Latent field Z(x) = (D(3), S(3), n(3), r, η, k, ρ, φ(2), g, u). Network g_θ: I_RGB → Z (with log-variances for n_xy, r, η, k, split D/S). Physics layer Φ: Z → Ŝ (Part J equations, η,k are scale factors on base constants η_c = n0_c·η, k_c = k0_c·k).
Identifiability: 3 observations per pixel vs ≥ 10 latent degrees of freedom ⇒ under-determined; the posterior over Z given I is multi-modal (e.g., diffuse vs specular attribution flips AoLP by ≈ π/2: on 32 synthetic environment-illumination views the untrained physics-only baseline has mean wrapped AoLP error 71° versus 9° when the true split is supplied; docs/SYNTHETIC_VALIDATION_REPORT.csv).
**CPE**: draw Z^(k) ~ p(Z | I) by perturbing raw latent logits with predicted variances while preserving D+S (so every sample reproduces the RGB), compute Î_θ^(k) = A(θ)Φ(Z^(k)); report mean, variance, empirical credible interval, DoLP interval, AoLP circular variance 1 − |mean e^{i2φ}|.

## Part L — PSRF [P] on [E]
R_p(θ) = a0 + a1 cos2θ + b1 sin2θ + a2 cos4θ + b2 sin4θ, fitted by least squares over the analyzer angles (pseudo-inverse of the harmonic basis). First order: a0 = S0/2, a1 = S1/2, b1 = S2/2 ⇒
DoLP = √(a1²+b1²)/a0, AoLP = ½ atan2(b1,a1). Order-2 identifiability requires ≥ 5 distinct angles mod π and a full-rank basis (4 angles are rank-deficient: sin4θ ≡ 0 at 0/45/90/135°) — enforced in code.
Model selection (hold-out-angle RMSE, BIC) decides whether 4θ terms are kept.

## Part M — Corrosion-feature fusion [P]
F_p = [R,G,B, L*,a*,b*, H,S,V, x,y, texture, roughness, specular/diffuse probability, DoLP_hat, cos2AoLP_hat, sin2AoLP_hat, a0,a1,b1, glare, uncertainty]; AoLP is **never** a scalar network input (circular).
Hue is encoded (cosH, sinH). CIELAB with D65 (Part M.1): XYZ = M_sRGB lin, f(t) = t^{1/3} or t/(3δ²) + 4/29, δ = 6/29; L* = 116 f(Y/Yn) − 16, a* = 500(f(X/Xn) − f(Y/Yn)), b* = 200(f(Y/Yn) − f(Z/Zn)).
CIEDE2000 (Sharma et al. 2005) as an auxiliary perceptual distance to a healthy-metal reference, not as chemical evidence. Fusion variants: early (concat inputs), mid (per-scale fusion with ECA), late (logit fusion).

## Part N — Objective [P]
L = λ_seg L_seg + λ_cls L_cls + λ_rec L_rec + λ_pol L_pol + λ_per L_periodic + λ_edge L_edge + λ_phys L_phys + λ_unc L_unc (+ λ_pit L_pit + λ_dist L_distill).
* L_seg = focal-CE + Dice; L_edge = BCE(‖∇p‖, ‖∇y‖) on soft labels; L_cls = ordinal BCE + Smooth-L1 on severity; L_pit = weighted BCE on the pit head.
* L_rec = ‖D + S − I_lin‖₁ + ½(‖∇ₓ·‖₁ + ‖∇ᵧ·‖₁) (the latent must explain the observation).
* L_pol (only if reference Stokes exist): L1(Ŝ0,S0) + L1(Ŝ_pol/Ŝ0, S_pol/S0) + circular term ⟨(1 − cos2Δφ) w⟩/⟨w⟩, cos2Δφ = (Ŝ1S1 + Ŝ2S2)/(‖Ŝ_{12}‖‖S_{12}‖), w = DoLP_ref (AoLP undefined at DoLP→0).
* L_periodic = ‖I_θ − I_{θ+π}‖² (+ ‖I_θ + I_{θ+π/2} − S0‖²). **Honest note:** for the closed-form Stokes analyzer both terms are identically zero (they are *tests*); the loss becomes active only for unconstrained analyzer-response heads (ablation arm that predicts R_p(θ) directly, used for H2).
* L_phys = ‖relu(√(Ŝ1²+Ŝ2²) − Ŝ0)‖² + ‖relu(−I_θ)‖² + (‖n‖−1)² + energy bound + TV on r, η, k.
* L_unc = ½e^{−s}·CE + ½s (aleatoric, s = log-variance); epistemic via MC-dropout/ensembles.
* L_distill = L_pol(Ŝ, S_meas) + MSE(feat_proj, feat_teacher) + T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)).
Wrapped AoLP distance: Δφ = min(d, π − d), d = (φ1 − φ2) mod π ∈ [0, π/2].

## Part O — Uncertainty [P]/[E]
Predictive uncertainty = aleatoric (heteroscedastic head) + epistemic (MC dropout mutual information, ensembles) + *ambiguity* (CPE spread of Î_θ and of corrosion evidence).
Calibration: ECE = Σ_b (|B_b|/N)|acc(B_b) − conf(B_b)|; Brier = mean Σ_k (p_k − 1[y=k])²; NLL = −mean log p_y; risk–coverage curve R(c) = error rate among the c-fraction of
lowest-uncertainty pixels; AURC = ∫R dc. Abstention: drop pixels above an uncertainty threshold chosen on validation data.
