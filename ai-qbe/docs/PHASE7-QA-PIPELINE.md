# AI-QBE — Phase 7: Quality Assurance Pipeline

Status: **Implemented and tested (274/274 tests passing).** This is the
phase the master prompt marks as mandatory before production use, and the
one where "never trust the generator's own claim" gets enforced
structurally rather than by convention.

## 1. What was implemented

- **`backend/validation/deterministic_math.py`** — best-effort SymPy
  verification for numeric MCQs: extracts a clean bare-arithmetic
  expression or single-variable equation from the question stem (handling
  implicit multiplication like `2x` and `^` exponent notation) and, when
  every option is a bare number, independently computes the answer and
  compares it to the claimed correct option. Deliberately narrow —
  `applicable=False` for anything else (conceptual questions, multi-step
  word problems, multi-variable systems) rather than guessing.
- **`backend/validation/answer_verification.py`** — the independent
  answer-verification stage (master prompt sections 21-22): tries the
  deterministic path first (authoritative when it applies, no LLM call
  needed), otherwise runs a genuinely separate LLM prompt — never the
  generation prompt reused — asking the model to derive the answer itself
  from the same evidence and return PASS/FAIL/UNCERTAIN via the
  `verification_schema.json` built in Phase 1. `UNCERTAIN` is a legitimate
  outcome that still reaches human review, never silently upgraded to a
  pass.
- **`backend/validation/distractor.py`** — near-duplicate option
  detection (embedding cosine similarity, when an embedding provider is
  available), a "possible second correct answer" flag (an incorrect
  option suspiciously close to the correct one), and length-outlier
  distractors (a common test-taking tell). Exact-duplicate options and
  raw option-length limits stay in structural validation from Phase 2;
  this module covers what that can't.
- **`backend/validation/dedup.py`** — the full 4-level semantic
  deduplication master prompt section 23 calls for, cheapest-first:
  exact normalized-hash → lexical similarity (`difflib`) → embedding
  cosine similarity against every still-alive candidate ever generated
  for the same subject/topic (across *all* jobs, not just the current
  one — a growing question bank shouldn't accumulate near-duplicates just
  because they came from different runs) → an LLM-judge tie-break, fired
  only for the borderline similarity band (0.80–0.92) so an LLM call
  never happens for the clear-cut cases levels 1–3 already resolved.
- **`backend/validation/difficulty_bloom.py`** — a checklist-style
  heuristic (calculation presence, sentence count, numeric-option ratio —
  not sentence length alone, per master prompt section 11) that estimates
  Bloom level and difficulty from the stem and cross-checks them against
  the planner's declared labels. A mismatch is recorded, never silently
  overridden — the reviewer's existing `change_difficulty`/`change_bloom`
  actions (Phase 2) are exactly what handle a real disagreement.
- **`backend/validation/quality_score.py`** — the composite score master
  prompt section 25 requires be "measurable," not the LLM's own
  self-reported confidence: every sub-score is computed from an upstream
  validator's actual output. A dedicated test parses the module's AST
  (with its docstring stripped) to prove `raw_item["confidence"]` never
  appears in the scoring code, not just that the tests happen to pass.
- **Wired into the executor** (`_apply_parsed_item`) as a strict pipeline
  after structural + notation validation: dedup → distractor →
  answer verification → difficulty/Bloom (informational) → quality
  scoring → final status. A `FAIL` verdict or a hard distractor failure
  now produces `REJECTED`; a caught duplicate produces `DUPLICATE`; a
  composite score below `settings.quality_score_low_confidence_threshold`
  (default 50) produces `LOW_CONFIDENCE` — none of these count toward a
  job's `requested_count`, so the over-generation loop from Phase 5
  correctly keeps working to close the real gap.
- **`GenerationJob.duplicate_count`** (defined since Phase 2, unpopulated
  until now) is computed for real at the end of each run.
- **A real bug found via MockProvider realism, not a contrived test**:
  MockProvider's original templated output (`"[MOCK #{index}] Which
  statement..."` with only a trailing counter varying) was lexically
  ~99% similar between consecutive items — every mock-generated batch
  was accidentally flagging nearly all of its own items as duplicates of
  each other. Fixed by rewriting `MockProvider._synthesize_mcq` to draw
  from a pool of 40 wholesale-distinct stem sentences and 5 distinct
  option banks (keyed by a process-wide, not per-instance, counter — so a
  freshly-constructed provider, as `regenerate` creates, never restarts
  at the same index and collides with an earlier candidate). This was a
  mock-realism bug the QA pipeline itself surfaced by actually working.
- **Tests** — 60 new tests, all passing (274 total): deterministic math
  verification (arithmetic, single/multi-solution equations, correct
  non-applicability), answer verification (deterministic path, LLM
  fallback, PASS/FAIL/UNCERTAIN all respected, context threading),
  distractor validation (length outliers, near-duplicates, possible
  second-correct-answer flag, graceful degradation with no embedding
  provider), deduplication (all 4 levels, including two dedicated tests
  that force the LLM-judge tie-break itself and check both possible
  verdicts are honored), difficulty/Bloom heuristics, quality scoring
  (including the AST-based "never uses self-reported confidence" test),
  and end-to-end executor integration tests (a deterministically-wrong
  answer rejected, a duplicate across two separate generation jobs for
  the same subject/topic blocked, an UNCERTAIN verdict still reaching
  review, a quality-score row persisted for every reviewed candidate) —
  plus an API-level test confirming quality scores and QA validation
  results are visible through the question endpoints.

```
$ python3 -m pytest tests/ -q
........................................................................
........................................................................
........................................................................
..........................................................
274 passed in 32.39s
```

## 2. Design decisions

- **Deterministic math verification is intentionally narrow.** Deriving
  "the correct answer" for an arbitrary generated question from nothing
  would require understanding its full semantic intent — a much larger
  NLP undertaking. What's implemented instead mirrors what Phase 1's
  benchmark task set already demonstrated works well: bare arithmetic and
  clean single-variable equations extracted directly from the stem. Ambiguous or
  multi-step word problems correctly fall through to the LLM verifier
  rather than being force-fit into an unreliable extraction.
- **Dedup is cheapest-check-first, and the LLM-judge tier is genuinely
  gated, not decorative.** A dedicated test (`test_llm_judge_only_invoked_
  for_borderline_band`) proves the LLM is never called for a clearly
  non-duplicate pair, and two more tests force a real borderline case and
  check that both a "yes" and "no" judge verdict are actually honored —
  this wasn't just built and assumed to work.
- **The dedup comparison pool spans every job for the same subject/topic,
  not just the current one.** A question bank grows across many
  generation runs over time; comparing only within one job's own output
  would miss the far more common real-world case of two separate runs
  months apart both generating "What is the SI unit of force?" A
  dedicated test exercises exactly this cross-job scenario.
- **LOW_CONFIDENCE is a distinct terminal status, not folded into
  REJECTED.** A candidate scoring below the quality threshold has already
  passed structural validation, notation checks, dedup, distractor
  validation, and didn't fail answer verification — its content may well
  be salvageable by a reviewer. Rejecting it outright would discard
  information; PENDING_REVIEW would silently dilute the main review queue
  with lower-confidence items. A separate status keeps both facts visible
  and query-able.
- **MockProvider's realism bug is documented as a finding, not smoothed
  over.** The near-99%-lexical-similarity problem was discovered because
  the new dedup logic actually ran against real (if mocked) generated
  content and correctly flagged what looked like near-duplicates. That
  the fix belongs in the mock rather than the dedup logic was verified by
  reasoning about what real distinct questions actually look like, not by
  loosening the threshold until tests passed.
- **The "multiple correct answers" executor test turned out to already be
  caught by structural validation's exact-duplicate-option check**
  (Phase 2), not the new distractor validator, once the
  `HashingEmbeddingProvider` limitation from Phase 3 was accounted for
  (two textually-identical options are an exact structural duplicate;
  two merely semantically-similar-but-differently-worded options aren't
  reliably close under a lexical, non-semantic embedding provider). The
  test and its docstring were adjusted to state which layer actually
  catches it, rather than asserting a specific validator without
  verifying that's really the one that fired.

## 3. Configuration

New setting (prefixed `AIQBE_`, see `backend/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `AIQBE_QUALITY_SCORE_LOW_CONFIDENCE_THRESHOLD` | `50.0` | Composite score (0-100) below which a structurally-valid, non-duplicate, non-rejected candidate is marked `LOW_CONFIDENCE` instead of `PENDING_REVIEW`. |

No new external dependencies — everything is built on `sympy` (already a
dependency since Phase 1), `difflib` (standard library), and the existing
`IEmbeddingProvider`/`ILLMProvider` abstractions.

## 4. Limitations

- **Answer verification still runs on the same `provider` as generation**,
  not a separate, dedicated verifier model. Phase 0 section 5 raises a
  generator/verifier split as worth evaluating once Phase 1's real
  benchmark exists; wiring a second provider through is a small, mostly
  mechanical change once that selection happens on real hardware — no
  architecture change needed.
- **Distractor and dedup embedding checks are only as good as the
  embedding provider supplied.** With the default `HashingEmbeddingProvider`
  (lexical, not semantic — a standing Phase 3 limitation), a paraphrase
  like "What is the SI unit of force?" vs. "Force is measured in which
  SI unit?" will NOT be caught by the embedding tier (cosine ~0.57 in
  practice, well under the dedup thresholds) — only word-overlapping
  near-duplicates are reliably caught until a real semantic model is
  wired in.
- **The Fe3+/NH4+-style ambiguity class of limitation recurs here in a
  different form**: `check_numeric_option_has_unit` and the difficulty/
  Bloom heuristics are keyword/pattern-based, not a learned classifier —
  by design (master prompt section 11 explicitly warns against estimating
  difficulty from surface features alone, but a checklist of features is
  still a checklist, not true pedagogical judgment).
- **No concept-coverage diversity metrics** (master prompt section 24: a
  topic's 500 questions covering only 3 of its 20 concepts) — that's a
  distinct, broader analysis across an entire approved bank rather than a
  per-candidate check, and is reasonable Phase 9 (large-scale generation
  tuning) scope once real volume exists to measure coverage against.
- **The LLM-judge dedup tie-break and the LLM-based answer verifier both
  currently run against `MockProvider`**, which honestly returns
  `UNCERTAIN`/`not-a-duplicate` rather than a real judgment — exactly the
  same "no real LLM yet" situation as every prior phase touching
  inference. Real duplicate-catch and verification accuracy rates don't
  exist until Phase 1's benchmark selects and runs a real model.
- **`quality_score_low_confidence_threshold` is a single global default**,
  not tuned per subject or difficulty. Real-world tuning (are physics
  numeric questions systematically scoring lower due to the units
  checker's false-positive rate, for instance) needs actual generation
  volume to observe against — reasonable Phase 9 work.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "All validators (structural, answer, distractor, notation, dedup,
> scoring) implemented and unit-tested; no candidate reaches APPROVED
> without passing every stage; quality score demonstrably derived from
> validator outputs (traceable, not opaque)."

**Met**: every validator listed is implemented, unit-tested in isolation,
and wired into the executor as a strict gate — a candidate can only reach
`PENDING_REVIEW` (still one step short of `APPROVED`, which remains a
human decision per Phase 2) after passing structural, notation, dedup,
distractor, and answer-verification checks, with the composite quality
score computed and persisted from exactly those validators' recorded
outputs (proven via the AST-inspection test, not just documentation
claiming it). What's explicitly not yet real: the LLM-backed pieces
(verifier, dedup judge) run against `MockProvider`, and embedding-based
checks are bounded by the lexical `HashingEmbeddingProvider` — both
already-documented, standing limitations from earlier phases, not new
gaps introduced here.
