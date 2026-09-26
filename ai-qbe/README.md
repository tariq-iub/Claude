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
  planner/executor (manual-context generation), structural validation,
  RBAC/auth, Celery job wiring, and Alembic migrations.
- [`docs/PHASE3-RAG.md`](docs/PHASE3-RAG.md) — the Phase 3 document RAG
  pipeline: embedding provider abstraction, Qdrant vector store, PDF/text
  ingestion, semantic chunking, topic-scoped retrieval, Topic Knowledge
  Packs, and citation-tracked generation.
- [`docs/PHASE4-INTERNET-RESEARCH.md`](docs/PHASE4-INTERNET-RESEARCH.md) —
  the Phase 4 controlled Internet research pipeline: approved-domain
  policy (deny-by-default), sandboxed fetching, HTML sanitization,
  two-layer prompt-injection defense, and web-to-knowledge-pack ingestion.
- [`docs/PHASE5-GENERATION-ENGINE.md`](docs/PHASE5-GENERATION-ENGINE.md) —
  the Phase 5 MCQ generation engine: the full topic × difficulty × Bloom ×
  question-type planner, real LLM-call batching, and the over-generation
  round loop that chases a job's requested approved-eligible count.
- [`docs/PHASE6-SCIENTIFIC-CONTENT.md`](docs/PHASE6-SCIENTIFIC-CONTENT.md) —
  the Phase 6 scientific notation pipeline: real MathJax-backed LaTeX
  validation (via a Node subprocess), chemistry formula/equation-balance
  checking, physics unit-presence flags, and the generation-time
  notation gate that blocks malformed candidates before human review.
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible / mock backends) and the JSON Schemas
  generation and verification calls must satisfy.
- [`backend/`](backend) — the FastAPI application: database models &
  migrations, the academic-data adapter, embedding provider abstraction,
  the generation planner/executor (manual-context and RAG modes, batched
  generation, over-generation rounds), structural validation, security
  (auth/RBAC/audit), REST API routers, and Celery worker tasks.
- [`rag/`](rag) — document extraction (PDF/text), semantic chunking, the
  vector-store abstraction (Qdrant), the ingestion pipeline, Topic
  Knowledge Pack retrieval, and (`rag/web/`) approved-domain search,
  fetching, sanitization, and prompt-injection defense.
- [`tools/mathjax_validator/`](tools/mathjax_validator) — a Node.js
  subprocess (real MathJax, via `mathjax-full`) that renders LaTeX/mhchem
  snippets to detect malformed notation; see `npm install` instructions
  there before running notation-related tests.
- [`scripts/benchmark/`](scripts/benchmark) — the ~100-task benchmark set
  (MCQ generation, answer verification, JSON-compliance stress, SymPy-
  checkable math) plus the runnable harness and report template.
- [`tests/`](tests) — 215 passing tests covering the benchmark grading
  logic and fixtures, the planner's apportionment math (including the
  full topic × difficulty × Bloom × question-type split), structural and
  scientific-notation validation, batched generation (one-LLM-call-per-
  batch, multi-batch splitting, item distinctness), over-generation round
  escalation and exhaustion, end-to-end generation-job execution
  (manual-context and RAG modes), a full API integration flow (auth/RBAC,
  job lifecycle, review actions, versioning, regenerate, document upload,
  RAG-grounded generation with citations), a real Alembic upgrade/
  downgrade round-trip, real PDF extraction, semantic chunking, real
  Qdrant vector-store behavior, domain-policy enforcement, HTML
  sanitization, prompt-injection scrubbing, a full web-ingestion flow
  against a simulated malicious page, and a seeded valid/malformed
  notation set run against real MathJax with zero false positives/negatives.

## Status

**Phase 6 — Scientific Content Support.** Implemented and tested
(`python3 -m pytest ai-qbe/tests -q` → 215 passed). Generated MCQs
containing math, physics, or chemistry notation are now checked against
**real MathJax** (a Node subprocess running `mathjax-full` — the same
library that will render questions in the eventual review UI, not an
approximation) plus independent chemistry-content checks (element-symbol
validity, reaction-equation atom-balance — catching things like an
unbalanced `H2 + O2 -> H2O` that valid-but-wrong LaTeX would sail through)
and a soft physics-unit-presence check. A candidate with malformed
notation is now blocked from reaching human review at generation time,
with the specific reason recorded for audit.

This is the first phase where "renders correctly" is verified against the
real target technology rather than deferred pending hardware/network
access — Node and MathJax need no external network access to check
notation once installed, unlike an LLM, an embedding model, or a live web
fetch. Those three remain unreal for the reasons stated in earlier
phases' notes: this sandbox's network policy blocks arbitrary outbound
requests, confirmed directly against multiple hosts (huggingface.co,
ollama.com, example.com). Also correctly out of scope until later phases:
independent fact-verification / deduplication / quality scoring /
concept-coverage diversity metrics (Phase 7) — which is why nothing is
auto-approved yet.
