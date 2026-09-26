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
- [`docs/PHASE7-QA-PIPELINE.md`](docs/PHASE7-QA-PIPELINE.md) — the Phase 7
  quality assurance pipeline: SymPy/LLM independent answer verification,
  distractor quality checks, 4-level semantic deduplication (including a
  gated LLM-judge tie-break), difficulty/Bloom cross-checking, and a
  composite quality score derived only from validator outputs.
- [`llm/`](llm) — the model-independent `ILLMProvider` interface (Ollama /
  llama.cpp / OpenAI-compatible / mock backends) and the JSON Schemas
  generation and verification calls must satisfy.
- [`backend/`](backend) — the FastAPI application: database models &
  migrations, the academic-data adapter, embedding provider abstraction,
  the generation planner/executor (manual-context and RAG modes, batched
  generation, over-generation rounds, the full Phase 7 QA pipeline),
  structural/notation/answer/distractor/dedup validation, security
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
- [`tests/`](tests) — 274 passing tests covering the benchmark grading
  logic and fixtures, the planner's apportionment math (including the
  full topic × difficulty × Bloom × question-type split), structural,
  scientific-notation, and full QA-pipeline validation, batched
  generation (one-LLM-call-per-batch, multi-batch splitting, item
  distinctness), over-generation round escalation and exhaustion,
  end-to-end generation-job execution (manual-context and RAG modes,
  independent answer verification, cross-job deduplication), a full API
  integration flow (auth/RBAC, job lifecycle, review actions, versioning,
  regenerate, document upload, RAG-grounded generation with citations,
  quality scores and QA results exposed through the review endpoints), a
  real Alembic upgrade/downgrade round-trip, real PDF extraction,
  semantic chunking, real Qdrant vector-store behavior, domain-policy
  enforcement, HTML sanitization, prompt-injection scrubbing, a full
  web-ingestion flow against a simulated malicious page, and a seeded
  valid/malformed notation set run against real MathJax with zero false
  positives/negatives.

## Status

**Phase 7 — Quality Assurance Pipeline.** Implemented and tested
(`python3 -m pytest ai-qbe/tests -q` → 274 passed). This is the phase the
master prompt marks mandatory before production use. Every generated
candidate now runs through independent answer verification (SymPy for
clean computable questions, an LLM verifier pass otherwise — never the
generator's own claimed answer taken on faith), distractor quality checks
(near-duplicate options, a "possible second correct answer" flag, length
outliers), 4-level semantic deduplication (exact hash → lexical →
embedding → a genuinely-gated LLM-judge tie-break for the borderline
band, compared against every still-alive candidate ever generated for the
same subject/topic across all jobs — not just the current one), a
difficulty/Bloom cross-check, and a composite quality score derived
solely from those validators' outputs (verified by an AST-inspection
test, not just asserted in a docstring). A candidate can now land at
`PENDING_REVIEW`, `REJECTED`, `DUPLICATE`, or `LOW_CONFIDENCE` — only the
first counts toward a job's requested target, so Phase 5's over-generation
loop now compensates for real rejection reasons, not just structural
ones. Building this surfaced and fixed a real bug: `MockProvider`'s
templated output was ~99% lexically similar between consecutive items,
so the new dedup logic correctly flagged nearly everything it generated
as duplicates of itself — fixed by giving the mock 40 wholesale-distinct
question stems instead of one template with a trailing counter.

Still unreal for the same standing reasons as every earlier phase
touching inference: no real LLM (verification and the dedup judge both
run against `MockProvider`) and no real semantic embedding model
(distractor/dedup embedding checks are bounded by the lexical
`HashingEmbeddingProvider` from Phase 3 — a paraphrase-level duplicate
like "What is the SI unit of force?" vs. "Force is measured in which SI
unit?" won't be caught until a real semantic model is wired in). Also
correctly out of scope until later phases: concept-coverage diversity
metrics across an entire approved bank (Phase 9) and the human review UI
itself (Phase 8) — which is why nothing is auto-approved yet, even though
everything needed to inform that decision is now computed and stored.
