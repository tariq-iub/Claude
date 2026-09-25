# INCF — Implicit Neural Corrosion Field

**1. Name/acronym:** Implicit Neural Corrosion Field (INCF)

**2. Core hypothesis:** Corrosion boundaries are physical, continuous
curves, not artefacts of a discrete pixel grid; representing the
segmentation as a **continuous function F: R² → R^C** (queryable at any
real-valued coordinate, conditioned on locally sampled encoder features)
should give sharper, resolution-independent boundaries than any fixed-grid
decoder (transposed conv / bilinear upsample of already-discretised
logits).

**3. Why fundamentally different:** Not SAM-style promptable segmentation
(no prompt encoder, no mask decoder trained on discrete tokens) and not a
generic NeRF-for-segmentation copy: the field is conditioned on **both** a
bilinearly-sampled CNN latent **and** the bilinearly-sampled raw Lab value
at the query coordinate, and uses **SIREN sinusoidal activations** so the
field itself has well-behaved, non-vanishing derivatives everywhere,
enabling stable arbitrary-resolution super-sampling of the mask at
inference — a capability a discrete decoder cannot provide without
retraining.

**4. Input representation:** `[R,G,B,L*,a*,b*,H,S,V]` (9 ch) at native
resolution, encoded to a `H/4×W/4` latent grid.

**5. Feature encoding:** strided-conv `LatentEncoder` → 24-d latent grid.

**6. Major blocks:** `LatentEncoder` → coordinate grid construction →
`grid_sample` (bilinear) of latent + Lab at query coordinates →
`ImplicitField` (3-layer SIREN MLP) → per-query logits.

**7. Information flow:** `RGB→Lab,HSV → encoder → latent grid → [query
coords ⊕ sampled latent ⊕ sampled Lab] → SIREN MLP → logits at queried
resolution (native or super-resolved)`.

**8. Mathematical formulation:**
`F(x,y) = MLP_SIREN([x,y, z(x,y), Lab(x,y)])`, `z(x,y) =
GridSample(latent, (x,y))`; SIREN layer `sin(ω₀·W·h + b)`.

**9. Corrosion-specific bias:** local Lab conditioning directly injected at
every query point (not just at the coarse latent level) keeps the
implicit boundary tightly locked to the actual perceptual colour edge even
when queried far above the latent grid's native resolution.

**10. Colour usage:** raw Lab sampled at the exact query coordinate is a
first-class input to the field function itself — a direct, continuous use
of CIELAB as part of the coordinate-conditioning, distinct from ΔE00-based
losses or kernels used elsewhere.

**11. Losses:** CE + Dice + boundary at training resolution; optionally an
extra consistency loss between predictions at native and randomly
super-sampled query grids (encourages resolution-consistent fields).

**12. Uncertainty:** finite-difference estimate of `||∇F||` at each query
point (cheap since the field is explicit and differentiable) — high
gradient magnitude flags a boundary/uncertain region.

**13. Boundary refinement:** *is* the core mechanism — query a denser grid
(e.g. 2–4×) purely at inference for a crisper boundary at no retraining cost.

**14. Multiclass strategy:** the field's output dimension is `C`; unchanged
by resolution or class count.

**15. Training:** train at native resolution (matches label resolution);
AdamW, cosine LR.

**16. Inference:** query at native or higher resolution as needed; `infer.py`/
`model.py` expose a `query_hw` argument for this.

**17. Complexity:** `O(N_query·hidden²)` for the MLP body, independent of
image resolution beyond the encoder; the encoder itself is `O(H·W)`.

**18. Parameters:** ≈ 18 K (measured 17,902).

**19. FLOPs:** encoder ≈ 0.05 GFLOPs at 128×128 (4× downsample); field ≈
`N_query·hidden²·layers` ≈ 0.3 GFLOPs at native 128×128 query.

**20. Memory:** `O(N_query·hidden)` activations; scales with the *query*
count chosen at inference, not a fixed decoder size.

**21. Strengths:** genuinely resolution-independent inference; smooth,
differentiable boundaries; conceptually elegant point-wise formulation
useful for irregular/point-cloud-like corrosion inspection data too.

**22. Weaknesses:** the MLP field, evaluated per-pixel, is less
parameter-efficient per FLOP than a pure conv decoder at a *fixed*
resolution; needs more training epochs to reach the same accuracy as
purely convolutional heads at native resolution (theoretical expectation,
to be tested).

**23. Ablations:** with/without local Lab conditioning; SIREN vs. ReLU
MLP; latent downsample factor; query resolution at inference (1×, 2×, 4×).

**24. Baselines:** U-Net (fixed-resolution decoder baseline), DeepLabV3+.

**25. Statistical protocol:** shared 5-seed protocol; additionally report
mIoU as a function of inference query resolution (unique curve for this
architecture family).
