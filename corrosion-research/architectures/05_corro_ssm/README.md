# Corro-SSM — Perceptually-Gated State-Space Corrosion Model

**1. Name/acronym:** Corro-SSM

**2. Core hypothesis:** Long-range context in a corrosion image is best
propagated along **scan lines with a forget gate tied to a physical
quantity** — a strong colour jump (ΔE00) between consecutive scanned
pixels almost certainly marks a material-state transition, so the
recurrent state should be reset there; smooth colour regions should let
state flow freely.

**3. Why fundamentally different:** This is a **custom, minimal, diagonal
linear SSM** (not a re-implementation of Mamba/S4 — no selective-scan
hardware kernel, no learned-only input-dependent gating), whose *entire*
gating mechanism is the closed-form function `sigmoid(-ΔE00/κ)` of an
explicit, physically defined colour-difference metric between consecutive
scan positions. No prior SSM/attention vision model ties its gate to
CIEDE2000.

**4. Input representation:** `[R,G,B,L*,a*,b*,H,S,V]` (9 ch).

**5. Feature encoding:** 1×1 conv stem → 20-d feature field.

**6. Major blocks:** `SeparableSSMBlock × 2`, each running a row-wise then
column-wise `PerceptualGatedSSM1D` recurrence.

**7. Information flow:** `RGB→Lab,HSV → stem → [row-scan SSM → col-scan
SSM]×2 → 1×1 head → logits`.

**8. Mathematical formulation:**
`g_t = sigmoid(-ΔE00(Lab_t,Lab_{t-1})/κ)`,
`h_t = g_t·(A⊙h_{t-1}) + (1-g_t)·(B x_t)`, `y_t = C h_t + D x_t`,
with `A=sigmoid(log_a)∈(0,1)^D` (diagonal, guaranteed-stable decay) and
`κ` a learned temperature.

**9. Corrosion-specific bias:** encodes "material boundaries reset
context" directly into the recurrence, rather than hoping a generic
learned gate discovers this from data alone.

**10. Colour usage:** exact differentiable ΔE00 computed between every
consecutive pair along each scan — a *routing* use (gating) of ΔE00, executed
`O(H·W)` times per block (linear, not quadratic).

**11. Losses:** CE + Dice + boundary (shared).

**12. Uncertainty:** `1 - gate` accumulated along both scan directions as a
"boundary crossing frequency" map, usable as an uncertainty/attention prior.

**13. Boundary refinement:** implicit via the gate itself (state resets
sharply at genuine boundaries); the shared boundary loss further supervises it.

**14. Multiclass strategy:** standard softmax head, class-count-agnostic core.

**15. Training:** AdamW, cosine LR; note the recurrence is a Python loop
over sequence length — training on larger images benefits from a batched
parallel-scan reformulation (documented as future engineering work; the
provided implementation favours **clarity of the physical gating math**
over kernel-level speed).

**16. Inference:** single forward pass (two full scans per block).

**17. Complexity:** `O(H·W·D²)` per block (linear in pixel count, unlike
self-attention's `O((HW)²)`).

**18. Parameters:** ≈ 3.8 K (measured 3,770).

**19. FLOPs:** ≈ `H·W·D²·const` ≈ 0.2 GFLOPs at 128×128.

**20. Memory:** `O(H·W·D)` activations, no attention matrix ever
materialised.

**21. Strengths:** linear-complexity long-range context; physically
grounded gating; very small parameter count.

**22. Weaknesses:** sequential Python-loop recurrence is slow in wall-clock
time on GPU without a custom parallel-scan kernel (a known, explicitly
documented limitation, not hidden); purely row/column scanning can miss
diagonal-oriented boundaries relative to a full 2D receptive field.

**23. Ablations:** learned gate only (no ΔE00) vs. perceptual gate;
`κ` fixed vs. learned; number of blocks; row-only vs. row+column scanning.

**24. Baselines:** Mamba-style vision SSM (learned-gate-only ablation
serves as this internal baseline), lightweight Transformer, BiSeNetV2.

**25. Statistical protocol:** shared 5-seed protocol; report wall-clock
separately given the known sequential-loop caveat.
