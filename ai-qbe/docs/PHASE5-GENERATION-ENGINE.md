# AI-QBE — Phase 5: MCQ Generation Engine

Status: **Implemented and tested (141/141 tests passing).** Runs entirely
against `MockProvider` — no real LLM benchmark has selected a model yet
(Phase 1 is still pending real hardware), so throughput/quality numbers
below describe the *mechanism*, not measured performance.

## 1. What was implemented

- **Full topic × difficulty × Bloom × question-type planner**
  (`backend/generation/planner.py`) — `build_plan()` now cross-splits a
  job's requested count across three independent distributions (it took
  two before Phase 5), still via the same largest-remainder apportionment
  so totals always sum exactly. `question_type_distribution` defaults to
  100% `single_best_answer`, so any caller that doesn't pass it gets
  identical behavior to before.
- **`QuestionType` enum** (`backend/domain/enums.py`) — four values
  (`single_best_answer`, `scenario_based`, `negative`,
  `definition_recall`), deliberately orthogonal to `BloomLevel`: format
  diversity, not cognitive-level diversity, which Bloom already covers.
  Added to `GenerationJobCreate` as `question_type_distribution`
  (validated: keys subset of the enum, sums to ~1.0) and persisted on
  `GenerationJob` (new column, migration `7e63872385e9`) and on
  `MCQCandidate.question_type` (column already existed, unused until now).
- **Real batching** (`backend/generation/executor.py`) —
  `_generate_batch_for_cell()` requests up to
  `settings.generation_max_batch_size` (default 20) questions in a single
  LLM call via a dynamically-built `{"items": [...]}` wrapper schema
  (`build_batch_schema()`, built from the same `MCQ_SCHEMA` used for
  single-item validation, so the two schemas never drift apart). Each
  item in the response is still validated independently — a batch
  succeeding or failing as a whole is not the same as any individual item
  passing structural validation. A model honestly returning fewer items
  than asked (the prompt explicitly permits this over fabricating extras)
  is not treated as an error.
- **Over-generation rounds** (docs/PHASE0-DESIGN.md section 30) — round 0
  now actually uses `job.over_generation_factor` (previously stored on the
  schema but ignored by the executor): it asks for
  `ceil(requested_count * over_generation_factor)` up front. If the
  resulting `PENDING_REVIEW` count still falls short of `requested_count`,
  additional rounds run (scaling the next ask by the observed accept rate,
  floored at 0.2 so one unlucky round can't request an absurd multiple),
  up to `min(job.max_attempts, settings.generation_max_rounds_ceiling)`.
  Every round's generated/valid/accept-rate is recorded as a
  `GenerationMetric` row, plus a final `rounds_used` and `target_met`
  metric — this is the first real use of the `generation_metrics` table
  (defined since Phase 2, unpopulated until now).
- **Question-type-aware prompts** — each plan cell's `question_type` now
  produces a different instruction fragment (e.g. "Phrase the question so
  the student must identify the one option that is FALSE..." for
  `negative`), applied in both the batch path and the single-item
  `_generate_one()` path (used by regenerate).
- **Shared per-item persistence** — `_apply_parsed_item()` factors out the
  validate-and-persist-one-item logic so both the batch path and the
  single-item regenerate path apply identical structural validation and
  status transitions; `_new_candidate()` factors out row creation
  (including the new `question_type` field) the same way.
- **Bug found and fixed while testing regenerate**: the `regenerate`
  endpoint's `PlanCell` construction dropped the original candidate's
  `question_type`, silently resetting every regenerated question to
  `single_best_answer` regardless of what type was superseded. Caught by
  writing a regenerate test with a deliberately non-default question type
  (`negative`) instead of the default — a test using the default value
  would have passed either way and hidden the bug. Fixed in
  `backend/api/routers/questions.py`.
- **Tests** — `test_planner.py` (extended), `test_batch_generation.py`
  (new: schema wrapping, one-call-per-batch, multi-call splitting when a
  cell exceeds the batch cap, item distinctness, question-type tagging,
  graceful handling of a stingy provider), `test_generation_executor.py`
  (rewritten: over-generation factor behavior, round escalation and
  exhaustion with a permanently-broken provider), `test_api_generation_engine.py`
  (new: invalid question-type rejected, custom distribution reflected in
  generated candidates, regenerate preserving question_type). 13 new
  tests; 141 total, all passing.

```
$ python3 -m pytest tests/ -q
........................................................................
.....................................................................
141 passed in 18.96s
```

## 2. Design decisions

- **Question type is orthogonal to Bloom, not a re-encoding of it.**
  Overlapping the two dimensions (e.g. a "recall" question type
  duplicating Bloom's "remember") would make the joint distribution
  meaningless — a scenario-based question can legitimately be at any
  Bloom level. Keeping them independent is what makes the three-way
  planner split (four-way counting difficulty) actually add diversity
  rather than just re-deriving one axis from another.
- **Batch failures are recorded, not silently dropped.** When a whole
  batch response fails to parse, one `INVALID` candidate is still written
  recording the raw failure — provenance applies to failed attempts too,
  per the project's standing "never lose provenance" rule, and it also
  means a job's `generated_count`/`rejected_count` reflect what actually
  happened, not just successes.
- **The inner batch loop always consumes its requested slot
  (`remaining_in_cell -= batch_n`), regardless of outcome.** An earlier
  version decremented by however many candidates a batch actually
  produced, which meant a chronically failing call would spin
  indefinitely inside one cell. Fixed to a simple, always-terminating
  inner loop; true shortfall compensation is deliberately pushed up to
  the outer, job-wide over-generation round loop instead of an unbounded
  per-cell retry — matching Phase 0's target-vs-final-target design,
  which reasons about the whole job, not one cell.
- **Accept-rate scaling is floored at 0.2 for the next round's ask.** A
  single round producing zero valid candidates (accept_rate=0) would
  otherwise divide by zero or demand an absurd multiple next round; the
  floor keeps escalation bounded (at most 5× the raw shortfall per round)
  while `max_attempts` bounds the total number of rounds regardless.
- **The batch wrapper schema is built dynamically from `MCQ_SCHEMA`
  (`build_batch_schema()`), not hand-duplicated as a second static JSON
  file.** A previous phase's single-item schema and a hand-maintained
  batch schema could drift out of sync (e.g. a field added to one and
  forgotten in the other); building the wrapper at call time from the one
  source of truth makes that impossible by construction.
- **`MockProvider` was extended to detect the batch wrapper shape and
  return N varied items**, parsing the requested count from the executor's
  own prompt wording. This keeps `MockProvider` a faithful stand-in for
  what a real batch-capable model needs to do, rather than only ever
  exercising the single-item code path.

## 3. Configuration

New settings (all prefixed `AIQBE_`, see `backend/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `AIQBE_GENERATION_MAX_BATCH_SIZE` | `20` | Ceiling on items requested per LLM call (never a floor — a cell needing fewer just asks for fewer). |
| `AIQBE_GENERATION_MAX_ROUNDS_CEILING` | `10` | Hard ceiling on over-generation rounds regardless of a job's own `max_attempts`, guarding against a misconfigured job. |

`GenerationJobCreate.question_type_distribution` (API field) defaults to
`{"single_best_answer": 0.6, "scenario_based": 0.25, "negative": 0.10,
"definition_recall": 0.05}` — an administrator can override it per job.

## 4. Limitations

- **No real model has generated a single real question yet** — every
  number and behavior described here comes from `MockProvider`, which is
  100% structurally compliant and always distinct by construction. Real
  batch-compliance rates, real over-generation round counts, and real
  question-type-following behavior are unknown until Phase 1's benchmark
  runs on the target hardware and a real model is wired in via
  `GenerationJobCreate.model_provider_type`.
- **No concept-coverage / diversity metrics** (master prompt section 24:
  detecting a topic's 500 questions covering only 3 of its 20 concepts) —
  that requires the semantic deduplication and diversity infrastructure
  that's explicitly Phase 7 scope, not Phase 5's planner/batching work.
  Phase 5's `question_type_distribution` is format diversity, a
  deliberately narrower and different thing.
- **The over-generation round loop has no early-exit "this is clearly
  never going to work" heuristic** beyond `max_attempts`/the rounds
  ceiling — a permanently-broken provider will run the full number of
  configured rounds (each potentially requesting a large batch) before
  giving up, rather than detecting a 0% accept rate after round 1 and
  stopping early. This is a reasonable simplification for now (the
  ceiling bounds the cost) but a smarter stop-early rule is a plausible
  Phase 9 (large-scale generation tuning) refinement once real
  accept-rate data exists to tune against.
- **`min_quality_score`, concept weighting beyond topic weight, and
  instructor-defined importance/exam-blueprint-driven allocation**
  (docs/PHASE0-DESIGN.md section 9's optional extensions) are not
  implemented — the planner splits by explicit topic `weight` and the
  job's own distributions only.
- **Batch size is not adapted to the model's actual context window** —
  `generation_max_batch_size` is a single global default, not computed
  from `GenerationModel.context_window` per model. A model with a small
  context window and a large `option_count` could in principle be asked
  for more items than comfortably fits; tuning this per-model is
  reasonable Phase 9 (large-scale generation) work once real token-budget
  data exists.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "Planner correctly distributes a requested total across topic/Bloom/
> difficulty cells; batches generate schema-valid JSON at a measured
> success rate; option-count/length constraints enforced."

**Met** for the planner-correctness and constraint-enforcement parts
(tested directly, including the awkward-number largest-remainder stress
test and the existing option-length/duplicate-option structural
validators from Phase 2). **Not measurable yet** for "batches generate
schema-valid JSON at a measured success rate" against a real model — that
number doesn't exist until Phase 1's benchmark runs for real, the same
gap every prior phase involving inference has carried forward.
