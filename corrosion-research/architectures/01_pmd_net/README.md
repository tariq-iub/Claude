# PMD-Net — Perceptual Manifold Diffusion Network

## 1. Name / acronym
**Perceptual Manifold Diffusion Network (PMD-Net)**

## 2. Core scientific hypothesis
Corrosion regions form connected sets on a *joint spatial–perceptual
manifold*: pixels belonging to the same oxide are close in CIELAB colour
space **and** tend to be spatially local, while healthy/oxide boundaries
correspond to sharp discontinuities of that joint manifold. If a learned
**anisotropic reaction–diffusion process** is allowed to smooth a feature
field along directions of low joint (space, colour) distance while being
blocked across directions of high distance, the steady state of that
diffusion is itself close to the desired segmentation, and the *bandwidths*
governing "close" in each domain are the only quantities that need to be
learned.

## 3. Why this is fundamentally different from existing approaches
It is **not** attention: attention computes a softmax kernel over
content-derived queries/keys with a learned inner product; PMD-Net computes
a **closed-form anisotropic Gaussian kernel** whose two bandwidths
(spatial σ_s, perceptual σ_c) are the *only* learned parameters of the
kernel itself, and the kernel is applied as an **explicit iterative Euler
integrator of a diffusion PDE**, not a single learned mixing step. It is not
a bilateral filter pre/post-process either — the kernel bandwidths and the
per-step retention gate `η` are trained end-to-end jointly with the encoder
and the segmentation head by backpropagating through every diffusion step.

## 4. Complete input representation
Per pixel: `[R,G,B, L*,a*,b*, H,S]` (8 channels). Full CIELAB, HSV and
spatial coordinate grids are available via `common.colorspace` and used by
the kernel module even though the CNN stem consumes a subset.

## 5. Feature encoding mechanism
A 2-layer 3×3 conv stem (GroupNorm + GELU) lifts the 8-channel input to a
24-dim feature field `F⁰ ∈ R^{24×H×W}`.

## 6. Major architectural blocks
1. **Stem encoder** — local feature lift.
2. **LocalPerceptualKernel** — computes per-pixel, per-neighbour diffusion
   weights over a (2r+1)² window from spatial offsets and a *learned Lab
   affine metric* (calibrating Euclidean Lab distance toward CIEDE2000
   sensitivity, e.g. weighting a\*,b\* more than L\*).
3. **DiffusionStep × T** — explicit Euler update `F ← F + σ(η)(𝒲∘F − F)`.
4. **1×1 classification head.**

## 7. Information flow
`RGB → {Lab,HSV} → concat → stem → F⁰ → [kernel weights from Lab] → T×diffusion → 1×1 conv → logits (B,C,H,W)`.

## 8. Mathematical formulation
Spatial distance: `d_s(i,j) = ||p_i - p_j||₂` for `j` in a local window.
Perceptual distance surrogate: `d_c(i,j) = ||M·Lab_i - M·Lab_j||₂` where `M`
is a learned diagonal-ish affine map (a smooth, differentiable proxy for
ΔE00 — see ablation "with/without ΔE00 component", which replaces `d_c`
with the true CIEDE2000 in the loss, see §10).

Kernel: `w_ij = softmax_j( -d_s(i,j)²/2σ_s² - d_c(i,j)²/2σ_c² )`.

Diffusion step: `F_i^{t+1} = F_i^t + sigmoid(η) · (Σ_j w_ij F_j^t - F_i^t)`,
a discretisation of `∂F/∂t = η·(𝒟_w F - F)` — anisotropic heat diffusion
restricted to directions of low joint spatial-perceptual distance.

## 9. Corrosion-specific inductive bias
Oxide regions are colour-coherent and spatially contiguous; the diffusion
denoises within-class colour variation (shadows, dirt, specular noise)
while an *automatically calibrated* σ_c prevents diffusion across genuine
material-boundary colour jumps.

## 10. Use of CIELAB / HSV / ΔE00
- Lab drives the perceptual kernel directly (learned affine ΔE00 surrogate).
- HSV hue/saturation channels feed the stem as complementary illumination-
  robust cues.
- The *exact* differentiable CIEDE2000 (`common.colorspace.deltaE2000`) is
  available as a **supervisory signal**: an optional auxiliary loss
  `L_ΔE = |d_c(i,j) − ΔE00(Lab_i,Lab_j)|` calibrates the learned surrogate
  toward the true perceptual metric without paying its full O(window) cost
  at every diffusion step (ablated in §23).

## 11. Loss functions
`CombinedSegLoss` = CE + Dice + boundary-gradient loss (see
`common/losses.py`), optionally + `L_ΔE` calibration term (weight 0.1).

## 12. Uncertainty modeling
Per-pixel predictive entropy of the softmax output after the final
diffusion step; high entropy pixels correlate with regions where the two
bandwidths disagree (colour-ambiguous transition zones).

## 13. Boundary refinement
Implicit: the diffusion process itself is a boundary-preserving smoother
(anisotropic diffusion is the continuous-PDE analogue of edge-preserving
filtering); no separate boundary head is required, though the shared
`boundary_loss` still supervises gradient alignment.

## 14. Multiclass segmentation strategy
Standard 6-way softmax head; the diffusion is class-agnostic (acts on the
generic feature field) so it scales to any number of classes without
architectural change.

## 15. Training procedure
AdamW, cosine LR, `CombinedSegLoss`, mixed precision on GPU, 15–30 epochs
on 96×96 random synthetic crops (or real tiles once available). See
`train.py`.

## 16. Inference procedure
Single forward pass (no test-time iteration beyond the fixed `T` diffusion
steps baked into the graph). See `infer.py`. `T` can be reduced at
inference for a speed/accuracy trade-off since each step is a bounded
Lipschitz contraction toward the fixed point.

## 17. Expected computational complexity
`O(H·W·k²·D)` per diffusion step (`k=2r+1` window, `D` feature dim) via
`unfold` — linear in image size, no quadratic all-pairs attention.

## 18. Estimated parameter count
≈ 7 K parameters at `D=24, r=3, T=4` (measured: 7,221). Trivially fits the
"<0.1 M" tier.

## 19. Estimated FLOPs
≈ `H·W·(k²·D·T + D²·const)` — for 128×128, `D=24, k=7, T=4`: ≈ 0.9 GFLOPs.

## 20. Memory requirements
Dominated by the `unfold` patches tensor, `O(B·D·k²·H·W)`; for 128×128,
`D=24,k=7` ≈ 19 MB per sample in fp32 — comfortably within 2 GB VRAM even
at batch size 8–16.

## 21. Expected strengths
Illumination/shadow robustness (diffusion denoises intensity noise while
respecting colour boundaries), extreme parameter efficiency, CPU-viable.

## 22. Expected weaknesses / failure cases
Long, thin corrosion filaments thinner than the diffusion window radius may
be over-smoothed; performance is sensitive to the calibration of σ_c
against genuinely low-contrast oxide transitions (e.g. early cuprite on
bright copper).

## 23. Ablations required
RGB-only vs Lab-only vs Lab+HSV input; with/without learned ΔE00
calibration loss; window radius `r ∈ {1,3,5}`; steps `T ∈ {1,2,4,8}`;
fixed vs learned σ_s, σ_c; with/without spatial term (σ_s → ∞).

## 24. Baselines
Fast-SCNN, BiSeNetV2, U-Net, Attention U-Net, DeepLabV3+/MobileNet, GMM,
K-Means, SLIC, classical Lab+ΔE00 nearest-centroid classifier.

## 25. Statistical superiority protocol
5 seeds × 5-fold data splits, paired Wilcoxon signed-rank test on per-image
mIoU against each baseline, Bonferroni-corrected α = 0.05/`n_baselines`.

## Files
- `model.py` — `PMDNet` (`build_model()` factory).
- `train.py` / `infer.py` — thin wrappers around `common.engine`.
