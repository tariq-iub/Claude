# AI Academic Question Bank Engine (AI-QBE)

Self-hosted, privacy-preserving pipeline that turns a university's syllabus
into large, validated, syllabus-aligned MCQ question banks using a locally
hosted LLM, controlled RAG retrieval, deterministic validation, and human
academic review.

## Contents

- [`docs/PHASE0-DESIGN.md`](docs/PHASE0-DESIGN.md) — the full Phase 0 technical
  design: architecture, data flow, ERD, model/embedding/vector-store choices,
  RAG and generation architecture, validation architecture, math/physics/
  chemistry strategy, security architecture, REST API surface, project
  structure, deployment architecture, performance constraints, risks, roadmap,
  and phase-by-phase acceptance criteria.
- [`docs/schema.sql`](docs/schema.sql) — PostgreSQL DDL for the AI-QBE metadata
  schema described in the ERD (generation jobs, sources, chunks, candidates,
  validation results, reviews, versions — provenance-complete).
- [`docs/PHASE1-BENCHMARK.md`](docs/PHASE1-BENCHMARK.md) — the Phase 1 local
  LLM benchmark harness: what was built, why no benchmark numbers are
  claimed yet (this dev environment has no GPU/model runtime), and how to
  run it on the real target workstation.
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible backends) and the JSON Schemas generation
  and verification calls must satisfy.
- [`scripts/benchmark/`](scripts/benchmark) — the ~100-task benchmark set
  (MCQ generation, answer verification, JSON-compliance stress, SymPy-
  checkable math) plus the runnable harness and report template.
- [`tests/`](tests) — 36 passing tests covering the grading logic, the
  benchmark fixtures' own ground truth, and the harness's wiring.

## Status

**Phase 1 — Local LLM Benchmark.** Harness, task set, and automated grading
are implemented and unit-tested (`python3 -m pytest ai-qbe/tests -q` → 36
passed). No model has actually been benchmarked yet: this development
environment has no GPU, no Ollama/llama.cpp installation, and no network
access to model registries, so — per this project's own "never claim
unmeasured numbers" rule — none are reported. Running
`scripts/benchmark/run_benchmark.py` against the 5 shortlisted models on
the real 8GB-VRAM workstation is the next step before Phase 2 (core
database + API) begins.
