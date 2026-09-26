# AI-QBE — Phase 1: Local LLM Benchmark

Status: **Harness complete and verified; no model benchmark results yet.**

## 1. What was implemented

- `llm/providers/` — the model-independent `ILLMProvider` interface plus
  three concrete backends: `OllamaProvider`, `LlamaCppProvider`, and
  `OpenAICompatibleProvider`. Each exposes `generate_text()` and
  `generate_structured()`; `generate_structured()` has a documented
  override point for grammar-constrained decoding (used by
  `LlamaCppProvider` via a GBNF-conversion hook, currently a conservative
  stub — see limitations).
- `llm/schemas/mcq_schema.json` and `llm/schemas/verification_schema.json`
  — the strict JSON Schemas generation and verification calls must satisfy
  (draft-07, `additionalProperties: false`), matching the MCQ/verification
  contracts in `docs/PHASE0-DESIGN.md` sections 12 and 21.
- `scripts/benchmark/tasks/` — the ~100-task benchmark set required by the
  master prompt's Phase 1 instructions, split into:
  - `mcq_generation_tasks.json` (40) — real academic context snippets and
    generation instructions across physics, math, chemistry, biology, and
    CS, spanning all four Bloom levels and all three difficulty levels.
  - `verification_tasks.json` (30) — real MCQs with a ground-truth correct
    answer and a claimed answer that is correct in half the set and
    deliberately wrong in the other half, so a verifier's ability to
    *catch* errors (not just agree) is actually tested.
  - `json_stress_tasks.json` (20) — adversarial JSON-compliance cases:
    LaTeX-heavy stems, matrices, Greek letters, `\ce{}` chemistry notation,
    quote-escaping, near-empty context, and a strict
    `additionalProperties: false` stress test.
  - `math_computation_tasks.json` (10) — algebra/calculus problems with a
    SymPy-checkable expected result (derivative, integral, limit, solve,
    factor, linsolve), used to test the "don't trust LLM math" verification
    strategy from Phase 0 section 22.
- `scripts/benchmark/validators.py` — automated, deterministic grading:
  JSON Schema validation, option-conciseness (≤6 words), distractor-hygiene
  (duplicate detection), single-correct-answer range checking, verifier
  verdict accuracy (with partial credit for a safe `UNCERTAIN` abstention
  vs. zero credit for a confidently wrong verdict), and SymPy-based math
  answer checking.
- `scripts/benchmark/resource_monitor.py` — a background thread sampling
  `nvidia-smi` (GPU utilization/VRAM) and `psutil` (system RAM) once per
  second during a run; degrades honestly (reports "not measured", never a
  guess) when no GPU is present.
- `scripts/benchmark/run_benchmark.py` — the harness entry point: loads all
  four task files, drives them through a configured provider, grades every
  response, and writes a single `report.json` plus a resource-usage summary.
- `scripts/benchmark/report_template.md` — the human-fillable comparison
  report (automated numbers + an expert-scored academic-quality rubric),
  one per candidate model, to be diffed side by side for model selection.
- Tests (`tests/test_benchmark_validators.py`,
  `tests/test_benchmark_tasks.py`, `tests/test_run_benchmark_wiring.py`) —
  36 tests, all passing, covering the grading logic, the correctness of the
  task fixtures' own ground truth (re-derived independently with SymPy
  rather than trusted as authored), and the harness's end-to-end wiring
  against a fake provider.

## 2. Why no benchmark numbers appear in this document

This development environment has **no GPU** (`nvidia-smi` not found), **no
Ollama/llama.cpp installed**, and **no network access to model
registries** (confirmed: outbound requests to ollama.com are blocked by the
sandbox's proxy). Running the actual candidate models from
`docs/PHASE0-DESIGN.md` section 5 (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B,
Gemma-2-9B, Phi-3.5-mini) is therefore not possible from here.

Per the master prompt's explicit instruction — "do not claim performance,
accuracy or throughput numbers unless they have actually been benchmarked"
— none are claimed. What's delivered instead is a harness that is fully
implemented, unit-tested against its own logic (36/36 passing, see below),
and ready to run unmodified on the actual 8GB-GPU / 32GB-RAM target
workstation.

```
$ python3 -m pytest tests/ -q
....................................
36 passed in 0.86s
```

## 3. How to get real numbers (next concrete action)

Follow `scripts/benchmark/README.md` on the target workstation:

1. Install Ollama (or llama.cpp server) and pull the 5 shortlisted models.
2. `pip install -r scripts/benchmark/requirements.txt`
3. Run `run_benchmark.py` once per model — same task set, same schema,
   same grading code, so results are directly comparable.
4. Have a subject-matter reviewer score a sample of each model's generated
   MCQs against the rubric in `report_template.md` for the dimensions that
   can't be graded automatically (clarity, distractor plausibility,
   factual correctness of open-ended generations).
5. Fill in one `report_template.md` per model and compare.

## 4. Design decisions in this phase

- **Automated grading is deliberately narrow.** It only checks what's
  checkable without a human or a second LLM acting as judge: JSON
  structure, option word-count, exact-duplicate options, verifier-verdict
  agreement with a known ground truth, and SymPy-checkable math. Anything
  requiring judgment (is this distractor *plausible*, is this explanation
  *clear*) is explicitly routed to the human rubric — this mirrors the
  master prompt's ban on self-graded LLM confidence as a quality metric,
  applied to the benchmark harness itself.
- **The verification task set is adversarial by construction.** Half of
  the 30 tasks pair a question with a wrong claimed answer specifically so
  a model that just agrees with whatever it's shown scores badly — a
  verifier that can't catch an obviously wrong answer is worse than useless
  for the Phase 7 independent-answer-verification stage it's being
  evaluated for.
- **`UNCERTAIN` gets partial credit, not zero, when the ground truth is
  PASS/FAIL.** A verifier that honestly abstains rather than confidently
  asserting a wrong verdict is the safer failure mode the master prompt's
  verification design (section 21) is built around; scoring it as pure
  failure would perversely reward overconfidence.
- **`LlamaCppProvider`'s JSON-Schema → GBNF grammar conversion is a stub
  that returns `None`** rather than a partial/unsound grammar. A grammar
  that silently accepts invalid shapes would be worse than falling back to
  prompt-only enforcement (which is still validated downstream). Building
  the real converter is left to Phase 5, once the selected model's actual
  schema-compliance failure modes are known from this benchmark's results
  — building it speculatively now would risk optimizing against the wrong
  failure patterns.
- **Resource monitoring degrades honestly.** On a machine without
  `nvidia-smi` (like this development sandbox), the report says
  `"gpu_monitoring_available": false` rather than omitting the field or
  guessing — so a report can never be misread as claiming GPU numbers that
  weren't actually collected.

## 5. Limitations

- No real model has been run yet — see section 2. This document will be
  updated with an actual comparison table and model selection once Phase 1
  is executed on the target hardware.
- The GBNF grammar-constrained decoding path for `LlamaCppProvider` is not
  yet implemented (returns `None`, falling back to prompt-only JSON
  enforcement) — see design decision above.
- `psutil`/`nvidia-smi`-based resource sampling captures system-wide
  RAM/GPU usage, not per-process; on a shared/multi-tenant workstation this
  could overstate what the benchmark itself consumed. Fine for a
  single-purpose benchmark run on an otherwise-idle machine, called out
  here so it isn't silently assumed accurate on a busier machine.
- Academic-quality scoring (factual correctness, clarity, distractor
  plausibility) is manual by design (section 4) — there is no automated
  substitute for it in this phase, and building an "LLM-judge" grader is
  explicitly deferred until Phase 7's independent verification design is
  in place and can be evaluated for judge reliability rather than assumed.

## 6. Acceptance criteria check (from docs/PHASE0-DESIGN.md section 20)

> "≥3 quantized models benchmarked on the target 8GB GPU across the
> ~100-task set; a generation model and (candidate) verifier model selected
> **with measured numbers**, not assumptions."

**Not yet met** — this requires running on real hardware, which this
environment cannot do. The harness, task set, and grading are complete and
tested; executing them on the target workstation is the next step before
Phase 2 begins.
