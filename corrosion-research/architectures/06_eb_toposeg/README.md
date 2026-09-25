# EB-TopoSeg — Energy-Minimizing Boundary-Topology Segmentation Network

**1. Name/acronym:** EB-TopoSeg

**2. Core hypothesis:** A good segmentation is the **minimiser of a learned
energy functional**, not the raw output of a feed-forward soft-max; making
inference an explicit, differentiable optimisation process (unrolled
gradient descent) allows the network to encode *global consistency*
(smoothness, topology) as first-class energy terms rather than hoping a
purely feed-forward receptive field discovers them implicitly.

**3. Why fundamentally different:** Not CRF-as-RNN (which uses mean-field
inference on a discrete CRF with fixed message-passing update forms) and not
a plain U-Net+refinement head: EB-TopoSeg predicts a genuine **unary energy
field** and unrolls **explicit gradient descent on a continuous relaxed
label simplex**, with a **differentiable soft-morphological-opening
topology penalty** substituting for persistent homology — a lightweight,
GPU-friendly topology surrogate not used in the CRF-as-RNN / DeepLab-CRF
literature.

**4. Input representation:** `[R,G,B,L*,a*,b*,H,S,V]` (9 ch).

**5. Feature encoding:** 2-layer conv backbone → 24-d features → unary
energy head (`C` channels) + scalar smoothness-weight head.

**6. Major blocks:** `EnergyNet` (unary + pairwise-weight prediction) →
unrolled energy-descent loop (`unroll_steps`) → `soft_opening` topology
regulariser.

**7. Information flow:** `RGB→Lab,HSV → backbone → (unary U, weight w) →
s₀=softmax(U) → [s ← softmax(s − lr·∂E/∂s)]×K → log(s) = logits`.

**8. Mathematical formulation:**
`E(s) = Σ_i CE(U_i, s_i) + Σ_{i~j} w_ij·||s_i−s_j||²`;
`s_{t+1} = softmax(s_t − lr·∂E/∂s_t)`, `lr` learned;
topology penalty `L_topo = ||s − opening(s)||²`,
`opening(s) = dilate(erode(s))` via `-maxpool(-·)` then `maxpool(·)`.

**9. Corrosion-specific bias:** oxide regions are large, topologically
simple blobs; the opening penalty explicitly discourages single-pixel
speckle predictions inconsistent with how corrosion actually forms and
grows.

**10. Colour usage:** Lab/HSV are backbone inputs; the boundary-loss term
(shared, `common/losses.py`) implicitly rewards label-gradient alignment
with the true class-boundary; a stronger variant (ablation) replaces the
generic gradient boundary loss with an explicit `||∇s|| vs ||∇(ΔE00 field)||`
alignment term.

**11. Losses:** CE + Dice + boundary + `0.05·L_topo`.

**12. Uncertainty:** entropy of the converged `s_T`; additionally, the
*distance moved* during unrolled descent (`||s_T − s_0||`) is a novel,
architecture-specific uncertainty signal — pixels that moved far from the
naive unary prediction were "corrected" by global consistency and are less
certain.

**13. Boundary refinement:** the pairwise smoothness term with a *learned,
spatially varying* weight `w(x)` acts as an adaptive, content-aware
boundary-preserving smoother (low `w` at true edges, high `w` in flat
regions).

**14. Multiclass strategy:** softmax simplex projection at every unroll
step generalises directly to any class count.

**15. Training:** backpropagation through the unrolled descent (`K≈5`
steps); gradient clipping recommended since unrolled optimisation can
amplify gradients.

**16. Inference:** run the same unrolled loop; `K` can be increased at
inference beyond the training value for extra refinement (test-time
optimisation), since each step is a bounded descent move.

**17. Complexity:** `O(K·H·W·C)` for the energy descent — linear, no
quadratic pairwise term (approximated via a fixed 4-neighbourhood
Laplacian, not a dense CRF).

**18. Parameters:** ≈ 7.4 K (measured 7,448).

**19. FLOPs:** ≈ `K·H·W·C·const` ≈ 0.3 GFLOPs at 128×128, `K=5`.

**20. Memory:** `O(H·W·C)` per unroll step retained for backprop through
time — moderate but bounded by small `K`.

**21. Strengths:** explicit control over segmentation smoothness/topology;
test-time-adjustable refinement depth; principled uncertainty from
descent-trajectory movement.

**22. Weaknesses:** unrolled optimisation can be unstable with large
learned step sizes; the 4-neighbourhood Laplacian approximation is weaker
than a true dense CRF for long-range consistency.

**23. Ablations:** `K∈{1,3,5,10}`; with/without topology penalty;
learned vs. fixed step size; 4- vs. 8-neighbourhood pairwise term.

**24. Baselines:** DeepLabV3+ (as the literature's CRF-refinement
reference point), U-Net, Attention U-Net, BiSeNetV2.

**25. Statistical protocol:** shared 5-seed protocol; additionally report
island-count (connected-component count of small spurious regions) as a
topology-specific metric compared against baselines.
