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
- [`docs/PHASE2-CORE-API.md`](docs/PHASE2-CORE-API.md) — the Phase 2 core
  database + REST API: models, AcademicDataPort adapter, generation
  planner/executor (manual-context generation, no RAG yet), structural
  validation, RBAC/auth, Celery job wiring, and Alembic migrations.
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible / mock backends) and the JSON Schemas
  generation and verification calls must satisfy.
- [`backend/`](backend) — the FastAPI application: database models &
  migrations, the academic-data adapter, generation planner/executor,
  structural validation, security (auth/RBAC/audit), REST API routers, and
  Celery worker tasks.
- [`scripts/benchmark/`](scripts/benchmark) — the ~100-task benchmark set
  (MCQ generation, answer verification, JSON-compliance stress, SymPy-
  checkable math) plus the runnable harness and report template.
- [`tests/`](tests) — 54 passing tests covering the benchmark grading
  logic and fixtures, the planner's apportionment math, structural
  validation, end-to-end generation-job execution, a full API integration
  flow (auth/RBAC, job lifecycle, review actions, versioning), and a real
  Alembic upgrade/downgrade round-trip.

## Status

**Phase 2 — Core Database & API.** Implemented and tested
(`python3 -m pytest ai-qbe/tests -q` → 54 passed). Subjects/topics are
readable via the `AcademicDataPort` adapter (seeded demo data for now);
generation jobs run end-to-end using **manually-supplied context** (no RAG
yet — that's Phase 3) against a configured `ILLMProvider`; every candidate
carries full provenance and goes through structural validation before
landing at `PENDING_REVIEW` for human review (approve/reject/edit/
regenerate, all versioned and audited).

No real LLM has been benchmarked or deployed yet: this development
environment still has no GPU or model runtime, so Phase 2 defaults to a
`MockProvider` for generation and defers Phase 1's real model selection to
the actual 8GB-VRAM target workstation. Also correctly out of scope until
later phases: RAG (Phase 3), Internet retrieval (Phase 4), and independent
fact-verification / deduplication / quality scoring (Phase 7) — which is
why nothing is auto-approved yet.
