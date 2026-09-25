# CPEN — Corrosion Prototype Evolution Network

**1. Name/acronym:** Corrosion Prototype Evolution Network (CPEN)

**2. Core hypothesis:** Oxidation is a *trajectory* in a latent chemical-appearance
space (healthy → early oxide → mature oxide), not an unordered bag of
colour clusters. Representing each class as an **ordered sequence of
evolving prototypes** and classifying by nearest-prototype read-out lets the
model capture intra-class appearance drift (e.g. thin vs. thick cuprite)
explicitly.

**3. Why fundamentally different:** Ordinary prototype networks (e.g.
few-shot ProtoNet) use one unordered prototype per class from an episodic
mean. CPEN maintains **K ordered prototypes per class**, updated online via
VQ-style EMA (not gradient descent, not episodic means), plus a
*trajectory-smoothness regulariser* that treats prototypes as samples along
a discrete 1-manifold — a temporal/ordinal structure absent from
differentiable-clustering or standard prototype literature.

**4. Input representation:** `[R,G,B,L*,a*,b*,H,S,V]` (9 ch).

**5. Feature encoding:** 2-layer dilated conv encoder → 32-d pixel embedding `e_i`.

**6. Major blocks:** `PixelEncoder` → `PrototypeField` (bank `P ∈ R^{C×K×D}`,
EMA update, soft nearest-prototype logits) → (no separate head; logits come
directly from negative distances).

**7. Information flow:** `RGB → encoder → embeddings e_i → distance to all
C·K prototypes → per-class min-distance → softmax logits`.

**8. Mathematical formulation:**
`p(c|e_i) = softmax_c(-min_k ||e_i - P_{c,k}||² / τ)`;
EMA update `N_{c,k}←γN_{c,k}+(1-γ)n_{c,k}`, `m_{c,k}←γm_{c,k}+(1-γ)Σe_i`,
`P_{c,k}=m_{c,k}/N_{c,k}`; trajectory loss
`L_traj = Σ_c Σ_k ||P_{c,k+1}-P_{c,k}||²`.

**9. Corrosion-specific bias:** encodes the physical prior that oxide
appearance evolves continuously with exposure time/thickness rather than
occupying disconnected appearance islands.

**10. Colour usage:** Lab/HSV are the raw features the encoder embeds; ΔE00
is not used directly in this variant (ablation: replace Euclidean
prototype distance with ΔE00-based distance in Lab-projected embedding
subspace).

**11. Losses:** CE + Dice + boundary + `0.01·L_traj` (trajectory smoothness,
detached from the main graph since prototypes are EMA-updated, used purely
as a monitored regularity metric / optional soft constraint on encoder
outputs near prototypes).

**12. Uncertainty:** margin between the closest and second-closest
prototype distances per pixel, converted to a confidence score;
low-margin pixels are boundary/transition candidates.

**13. Boundary refinement:** none built-in; relies on the shared boundary
loss — a natural ablation target (add a boundary head).

**14. Multiclass strategy:** native — one bank of `K` prototypes per class,
class count only changes the bank's first dimension.

**15. Training:** AdamW on the encoder only (prototypes are non-gradient,
EMA γ=0.98); warm-start prototypes with k-means on first-batch embeddings
recommended for real data.

**16. Inference:** forward pass, prototypes frozen (no EMA update in eval mode).

**17. Complexity:** `O(H·W·C·K·D)` for the distance computation — linear in
pixels, small constant (`C·K` ≈ 18).

**18. Parameters:** ≈ 13 K (measured 13,056).

**19. FLOPs:** ≈ `H·W·C·K·D·2` ≈ 0.3 GFLOPs at 128×128.

**20. Memory:** negligible beyond activations; prototype bank is a few KB.

**21. Strengths:** naturally suited to *severity estimation* (prototype
index along the trajectory ≈ oxidation stage); interpretable (nearest
prototype is inspectable); strong small-dataset behaviour (few learnable
parameters in the classifier itself).

**22. Weaknesses:** sensitive to EMA decay/initialisation; can collapse
prototypes if `K` too large for available diversity; no explicit spatial
smoothing (relies entirely on per-pixel embedding quality).

**23. Ablations:** `K∈{1,2,3,5}`; EMA γ; with/without trajectory loss;
Euclidean vs ΔE00-based prototype distance; encoder capacity.

**24. Baselines:** classical Lab+ΔE00 nearest-centroid, GMM, K-Means,
DeepLabV3+/MobileNet, U-Net.

**25. Statistical protocol:** same 5-seed paired Wilcoxon protocol as
PMD-Net (see master `RESEARCH.md`).
