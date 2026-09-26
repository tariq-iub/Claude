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
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible / mock backends) and the JSON Schemas
  generation and verification calls must satisfy.
- [`backend/`](backend) — the FastAPI application: database models &
  migrations, the academic-data adapter, embedding provider abstraction,
  generation planner/executor (manual-context and RAG modes), structural
  validation, security (auth/RBAC/audit), REST API routers, and Celery
  worker tasks.
- [`rag/`](rag) — document extraction (PDF/text), semantic chunking, the
  vector-store abstraction (Qdrant), the ingestion pipeline, and Topic
  Knowledge Pack retrieval.
- [`scripts/benchmark/`](scripts/benchmark) — the ~100-task benchmark set
  (MCQ generation, answer verification, JSON-compliance stress, SymPy-
  checkable math) plus the runnable harness and report template.
- [`tests/`](tests) — 91 passing tests covering the benchmark grading
  logic and fixtures, the planner's apportionment math, structural
  validation, end-to-end generation-job execution (manual-context and RAG
  modes), a full API integration flow (auth/RBAC, job lifecycle, review
  actions, versioning, document upload, RAG-grounded generation with
  citations), a real Alembic upgrade/downgrade round-trip, real PDF
  extraction, semantic chunking, and real Qdrant vector-store behavior.

## Status

**Phase 3 — Document RAG.** Implemented and tested
(`python3 -m pytest ai-qbe/tests -q` → 91 passed). Documents (PDF/text/
Markdown) can be uploaded, cleaned, chunked, embedded, and stored in
Qdrant with full metadata; generation jobs can now run in **RAG mode**
(`use_rag: true`), retrieving a Topic Knowledge Pack from ingested
documents instead of requiring manually-supplied context, with every
resulting candidate recording exactly which source chunks it was grounded
in (`MCQSource`, surfaced as `source_chunk_ids` on the question API).
Manual-context mode (Phase 2) still works unchanged.

Two things remain intentionally unreal pending real hardware: **no LLM**
(generation still defaults to `MockProvider`, per Phase 1/2) and **no
semantic embedding model** (retrieval defaults to a dependency-free,
clearly-labeled lexical `HashingEmbeddingProvider`, since this sandbox's
network policy blocks downloading model weights from huggingface.co, the
same constraint that blocked Phase 1's LLM benchmarking). Both real
implementations are written to the same provider interfaces and are a
config change away once they can be run and benchmarked on the actual
8GB-VRAM workstation. Also correctly out of scope until later phases:
Internet retrieval (Phase 4) and independent fact-verification /
deduplication / quality scoring (Phase 7) — which is why nothing is
auto-approved yet.
