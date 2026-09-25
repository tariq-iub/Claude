# MoMER — Mixture-of-Material-Experts Router Network

**1. Name/acronym:** MoMER

**2. Core hypothesis:** Different corrosion appearances (thin cuprite film,
thick black tenorite crust, powdery nantokite, bright atacamite bloom) are
visually distinct enough that **specialised small experts** each handling
one regime should outperform one generalist network of equal total
capacity — provided the routing decision is anchored to an explicit colour
prior so specialisation is guided, not left to emerge unstably from
gradient noise alone.

**3. Why fundamentally different:** Not a standard sparse MoE (whose gate is
a pure learned function of hidden features): MoMER's gate is the **product
of a learned content logit and an explicit Gaussian colour-prior
likelihood** over learnable per-expert Lab centroids, i.e. a **hybrid
learned/parametric routing density**. Its uncertainty output is not a
bolted-on auxiliary head but literally the **Shannon entropy of the routing
distribution that produced the prediction** — the uncertainty and the
prediction share the same computational origin.

**4. Input representation:** `[R,G,B,L*,a*,b*,H,S,V]` (9 ch); a locally
averaged Lab statistic (5×5 box) used as the colour-prior query.

**5. Feature encoding:** 3×3 conv stem → 20-d shared feature field, fed to
every expert and the router.

**6. Major blocks:** `TinyExpert × E` (5 lightweight per-pixel conv
classifiers) + `ColorPriorRouter` (learned content gate ⊙ Gaussian colour
prior).

**7. Information flow:** `RGB→Lab,HSV → shared stem → {experts' logits,
router gate} → weighted sum → final logits`; router additionally exposes
per-pixel routing entropy.

**8. Mathematical formulation:**
`g_e(x) = softmax_e(z_e(x) + log N(L̄ab(x); μ_e, Σ_e))`;
`logits(x) = Σ_e g_e(x)·Expert_e(x)`;
`Uncertainty(x) = H(g(x))/log E = -Σ_e g_e log g_e / log E ∈ [0,1]`.

**9. Corrosion-specific bias:** the Gaussian colour prior is initialised
(and can be partly fixed) near the known mean Lab values of each oxide
chemistry, giving experts a physically motivated starting specialisation
rather than random symmetry breaking.

**10. Colour usage:** locally averaged Lab is the *direct query variable*
for a parametric (Gaussian) colour-likelihood term multiplied into the
routing softmax — colour is literally part of the routing probability
model, not merely a network input.

**11. Losses:** CE + Dice + boundary; optional load-balancing auxiliary
loss (encourage roughly uniform expert utilisation) as a standard MoE
regulariser, ablatable.

**12. Uncertainty:** native output — the routing entropy map, requiring no
Monte-Carlo sampling or ensembling.

**13. Boundary refinement:** boundaries naturally show elevated routing
entropy (color statistics are locally mixed there), which can be fed back
as an attention prior to the shared boundary loss (ablatable design).

**14. Multiclass strategy:** each expert is a full `C`-way classifier;
final output is the gate-weighted mixture, so class count changes only the
expert head width.

**15. Training:** AdamW; consider a small load-balancing loss to prevent
expert collapse; router and experts trained jointly end-to-end.

**16. Inference:** single forward pass; optionally hard-route to the
arg-max expert for speed (ablation: soft vs. hard routing accuracy/speed
trade-off).

**17. Complexity:** `O(E·H·W·D)` for all experts evaluated densely (dense
MoE here, not sparse top-k, since `E` is small and hardware is
memory-bound rather than compute-bound at this scale).

**18. Parameters:** ≈ 17 K (measured 16,965) for `E=5`.

**19. FLOPs:** ≈ `E·H·W·width·const` ≈ 0.4 GFLOPs at 128×128.

**20. Memory:** `O(E·H·W·C)` for stacked expert outputs — modest at small `E`.

**21. Strengths:** built-in, computationally free per-pixel uncertainty;
naturally interpretable (inspect which expert dominates where);
physically anchored specialisation aids small-dataset generalisation.

**22. Weaknesses:** dense evaluation of all experts wastes compute relative
to true sparse top-1 routing at larger `E`; Gaussian colour prior is a
simplification that may not separate oxides with overlapping Lab
distributions (e.g. early cuprite vs. healthy copper under warm lighting).

**23. Ablations:** `E∈{2,3,5,8}`; with/without colour prior (pure learned
gate); with/without load-balancing loss; hard vs. soft routing at inference.

**24. Baselines:** single dense classifier of equal total parameter count
(the natural ablation baseline for "does mixture help"), GMM (itself a
mixture model, a natural classical comparator), DeepLabV3+/MobileNet.

**25. Statistical protocol:** shared 5-seed protocol; additionally report
calibration error (ECE) specifically correlated against the routing-entropy
uncertainty map (a sanity check that the entropy signal is meaningful).
