# HyperCorNet — Hypergraph Corrosion Recognition Network

**1. Name/acronym:** HyperCorNet

**2. Core hypothesis:** Corrosion recognition benefits from **higher-order,
non-pairwise** relations: a single hyperedge can bind an entire homogeneous
oxide patch (local hyperedge) *and* separately bind all patches across the
image sharing the same oxide chemistry (global colour hyperedge), which a
pairwise graph (GNN) cannot represent without O(N²) edges.

**3. Why fundamentally different:** Not a GNN with attention, and not
SLIC-then-GNN as a two-stage pipeline: the superpixel assignment (soft
SLIC) is a *differentiable module inside the network*, trained jointly, and
the propagation operator is the **spectral hypergraph convolution**
`D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ` with a *second* incidence
matrix built from learned colour-prototype membership — a two-level
hypergraph (pixel→cell, cell→colour-bin) not present in prior hypergraph
segmentation work.

**4. Input representation:** `[R,G,B,L*,a*,b*,x,y]` (8 ch).

**5. Feature encoding:** 3×3 conv stem → 24-d pixel features.

**6. Major blocks:** `SoftSLIC` (differentiable soft superpixel assignment,
one Lloyd-style iteration) → `HypergraphConv × 2` (local + global colour
hyperedges) → 1×1 classification head.

**7. Information flow:** `RGB→Lab,xy → stem features → soft pixel-to-cell
incidence H → hypergraph conv (local+global) → scatter back to pixels →
logits`.

**8. Mathematical formulation:** incidence
`a_{i,m}=softmax_m(-d_s(i,m)²/σ_s² - d_c(i,m)²/σ_c²)`; hypergraph conv
`X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ`; global bins via second
soft incidence to learned Lab prototypes.

**9. Corrosion-specific bias:** disjoint patches of the *same* corrosion
type (e.g. two separate cuprite spots) should share statistical strength —
exactly what the global colour hyperedge provides, without full N² attention.

**10. Colour usage:** Lab distance drives both incidence matrices (spatial
+ chromatic for cells; pure chromatic similarity to learned bins for global
hyperedges).

**11. Losses:** CE + Dice + boundary (shared).

**12. Uncertainty:** entropy of the pixel→cell soft assignment as a
"superpixel-boundary" uncertainty proxy.

**13. Boundary refinement:** cell boundaries are inherently soft (assignment
is continuous), giving graded confidence at oxide transitions; combine with
shared boundary loss.

**14. Multiclass strategy:** standard softmax head on scattered-back pixel features.

**15. Training:** AdamW, cosine LR, joint end-to-end (SLIC bandwidths +
hypergraph weights + head).

**16. Inference:** one forward pass; grid resolution (`grid_h,grid_w`)
trades boundary fidelity for compute.

**17. Complexity:** `O(H·W·M)` for incidence (M = number of cells, ≪ H·W),
linear in image size.

**18. Parameters:** ≈ 4.4 K (measured 4,388).

**19. FLOPs:** ≈ `H·W·M·D` per hypergraph conv ≈ 0.1–0.3 GFLOPs at 128×128, `M=64`.

**20. Memory:** incidence matrices are `O(H·W·M)`, small for reasonable `M`.

**21. Strengths:** captures long-range, disconnected same-material regions;
naturally multi-scale via grid resolution; lightweight.

**22. Weaknesses:** grid-based cells may misalign with true irregular oxide
boundaries at very low `M`; global colour bins can conflate visually
similar but chemically distinct oxides without enough bins.

**23. Ablations:** grid size; number of colour bins; local-only vs
local+global hyperedges; learned vs fixed σ_s/σ_c.

**24. Baselines:** SLIC-based classical pipelines, GNN (pairwise) ablation,
DeepLabV3+/MobileNet, BiSeNetV2.

**25. Statistical protocol:** shared 5-seed paired-test protocol (see
`RESEARCH.md`).
