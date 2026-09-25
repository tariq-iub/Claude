# CorroNCA — Corrosion Neural Cellular Automaton

**1. Name/acronym:** CorroNCA

**2. Core hypothesis:** A correct segmentation is a **self-consistent local
fixed point**: if every cell (pixel) repeatedly asks "given my current
belief and my neighbours', should I update?", using only local
information, the whole grid should converge to a globally coherent
labelling — analogous to how physical corrosion itself propagates locally
from nucleation sites.

**3. Why fundamentally different:** This is not a shallow CNN with a few
extra layers; the **same tiny update rule is applied recursively at every
pixel for T iterations**, weight-shared across space and time (like
Mordvintsev-style growing NCA), repurposed here for *recognition* rather
than *pattern generation*, with a corrosion-specific perception channel
(local ΔE00 to each of the 4 neighbours) that no generative-NCA or standard
CNN perception filter includes.

**4. Input representation:** `[R,G,B,L*,a*,b*]` (6 ch) → lifted to a
hidden cell state (num_classes + hidden_dim channels).

**5. Feature encoding:** state channels *are* the features; the first
`num_classes` channels double as the (evolving) class-logit read-out.

**6. Major blocks:** `PerceptionFilter` (fixed depthwise Sobel/Laplacian +
learned-free ΔE00-to-neighbour channels) → tiny 1×1-conv `update_rule` MLP →
stochastic "fire-rate" masked residual update, iterated `T` times.

**7. Information flow:** `RGB,Lab → stem → state₀ → [perceive→update]×T →
state_T[:,:C] = logits`.

**8. Mathematical formulation:** perception
`P_i = [s_i, Sobel_x*s_i, Sobel_y*s_i, Laplace*s_i, ΔE00(i,left), ΔE00(i,right), ΔE00(i,up), ΔE00(i,down)]`;
update `s_i ← s_i + m_i·f_θ(P_i)`, `m_i ~ Bernoulli(fire\_rate)` (stochastic
async-like update during training only).

**9. Corrosion-specific bias:** explicit ΔE00-to-neighbour perception
channels directly encode "am I at a material boundary?" as a first-class
local signal driving the automaton's dynamics.

**10. Colour usage:** ΔE00 computed exactly (autograd-differentiable) between
each cell and its 4 orthogonal neighbours every iteration — a *routing/
perception* use of ΔE00, not merely a loss term.

**11. Losses:** CE + Dice + boundary on the final iteration's logits;
optionally an *overshoot* loss summing the same objective over the last few
iterations (encourages fast, stable convergence).

**12. Uncertainty:** variance of the per-pixel argmax class across the last
few iterations — a natural "settled vs. still-changing" uncertainty signal
unique to iterative dynamical models.

**13. Boundary refinement:** emergent from the automaton's own dynamics
(neighbourhood consensus); no separate module.

**14. Multiclass strategy:** first `num_classes` state channels are read out
directly as logits — changing class count only changes state width.

**15. Training:** stochastic fire-rate ("dropout in time") for robustness to
asynchronous updates; short BPTT through T≈12 steps; gradient clipping
recommended for stability.

**16. Inference:** deterministic (fire-rate=1), can early-stop once the
state stabilises (Δstate < ε) for an anytime/adaptive-compute model.

**17. Complexity:** `O(T·H·W·D)`, no attention, no pooling — extremely
cheap per step.

**18. Parameters:** ≈ 3.2 K (measured 3,184) — smallest of all 8 candidates.

**19. FLOPs:** ≈ `T·H·W·(D·3·D_perc)` ≈ 0.15 GFLOPs at 128×128, `T=12`.

**20. Memory:** only the current state tensor is kept (no skip connections
to store) — the lowest-memory architecture in this repository.

**21. Strengths:** smallest model, translation-equivariant by
construction, "anytime" inference (stop early on tight compute budgets),
naturally models corrosion *spread* (useful for a follow-up growth/
prognosis task).

**22. Weaknesses:** can be slow to propagate information across large
distances (local-only receptive field grows only linearly with T);
training can be unstable without careful fire-rate/step-size tuning.

**23. Ablations:** `T∈{4,8,12,24}`; with/without ΔE00 perception channels
(replace with plain RGB-difference); fire-rate; hidden_dim.

**24. Baselines:** Fast-SCNN, BiSeNetV2, classical GMM/K-Means, U-Net.

**25. Statistical protocol:** shared 5-seed protocol; additionally report
accuracy **as a function of T** (anytime accuracy curve) as a unique
experiment for this family.
