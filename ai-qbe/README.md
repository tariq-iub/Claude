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
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible / mock backends) and the JSON Schemas
  generation and verification calls must satisfy.
- [`backend/`](backend) — the FastAPI application: database models &
  migrations, the academic-data adapter, embedding provider abstraction,
  generation planner/executor (manual-context and RAG modes), structural
  validation, security (auth/RBAC/audit), REST API routers, and Celery
  worker tasks.
- [`rag/`](rag) — document extraction (PDF/text), semantic chunking, the
  vector-store abstraction (Qdrant), the ingestion pipeline, Topic
  Knowledge Pack retrieval, and (`rag/web/`) approved-domain search,
  fetching, sanitization, and prompt-injection defense.
- [`scripts/benchmark/`](scripts/benchmark) — the ~100-task benchmark set
  (MCQ generation, answer verification, JSON-compliance stress, SymPy-
  checkable math) plus the runnable harness and report template.
- [`tests/`](tests) — 128 passing tests covering the benchmark grading
  logic and fixtures, the planner's apportionment math, structural
  validation, end-to-end generation-job execution (manual-context and RAG
  modes), a full API integration flow (auth/RBAC, job lifecycle, review
  actions, versioning, document upload, RAG-grounded generation with
  citations), a real Alembic upgrade/downgrade round-trip, real PDF
  extraction, semantic chunking, real Qdrant vector-store behavior,
  domain-policy enforcement, HTML sanitization, prompt-injection
  scrubbing, and a full web-ingestion flow against a simulated malicious
  page.

## Status

**Phase 4 — Controlled Internet Research.** Implemented and tested
(`python3 -m pytest ai-qbe/tests -q` → 128 passed). Documents can now be
ingested from **approved-domain URLs** as well as direct upload: an
administrator manages an explicit approved/blocked domain list
(deny-by-default — nothing is ingestible until approved), and generation
jobs' RAG mode automatically picks up web-sourced evidence alongside
uploaded documents, since both land in the same chunk store. Every fetched
page is sanitized (scripts/styles/nav/footer/comments stripped) and passed
through a prompt-injection scrub before storage, with a second independent
defense layer (an explicit "this is data, not instructions" wrapper) now
applied to every generation prompt.

Three things remain intentionally unreal pending real hardware/network
access, all for the same reason — this sandbox's network policy blocks
arbitrary outbound requests (confirmed directly against multiple hosts):
**no LLM** (generation still defaults to `MockProvider`), **no semantic
embedding model** (retrieval still defaults to a clearly-labeled lexical
`HashingEmbeddingProvider`), and **no live web fetch** (the real
`RequestsFetcher` is implemented and unit-tested via a mocked transport,
but not exercised against the actual Internet). All three are one config
change away from their real implementations once run on the target
workstation with normal network access. Also correctly out of scope until
later phases: independent fact-verification / deduplication / quality
scoring (Phase 7) — which is why nothing is auto-approved yet.
