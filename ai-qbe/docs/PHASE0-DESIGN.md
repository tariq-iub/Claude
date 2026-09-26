# AI-QBE — Phase 0: Requirements & Architecture

Status: **DRAFT FOR REVIEW**. No implementation exists yet. All performance
numbers below are *targets or estimates*, not measurements — Phase 1
produces the first real benchmark data, and this document is updated once
that data exists.

---

## 0. Scope of this document

This is Phase 0 only: architecture and design. It does not contain
application code. It exists to get explicit sign-off on technology choices,
data model, and phase boundaries before any implementation work (Phase 1+)
starts, per the mandatory incremental-development instruction.

MVP / Production / Future are distinguished throughout using these tags:

- **[MVP]** — required to demonstrate the end-to-end pipeline (Phases 1–9).
- **[PROD]** — required before real university deployment (Phase 11).
- **[FUTURE]** — explicitly deferred; schema/architecture must not block it.

---

## 1. Recommended Technology Stack

| Layer | Choice | Why |
|---|---|---|
| API/backend | **Python 3.11 + FastAPI** | async-first, native Pydantic → clean JSON-schema validation of LLM output, huge RAG/ML ecosystem (transformers, sentence-transformers, sympy) lives in Python. |
| Background jobs | **Celery + Redis** (broker + result backend) | mature, supports retries/resumability, and Redis doubles as a cache. RQ was considered — rejected, weaker support for long-running chained pipelines and revocation semantics. |
| Primary DB | **PostgreSQL 15+** | JSONB for flexible LLM-output fields, strong FK/constraint support for provenance chains, mature migration tooling (Alembic), pgvector available as a fallback vector store. |
| Existing academic DB | **Adapter layer**, read-mostly, dialect-agnostic (SQLAlchemy Core + a `AcademicDataPort` interface); first target dialect **MySQL/MariaDB** since that's the most common existing university SIS backing store, but never hard-coded. | Section 5 requires we never assume ownership of the university schema. |
| Vector store | **Qdrant** (self-hosted, single binary/Docker) — see §7 for the pgvector-vs-Qdrant tradeoff | metadata filtering + ANN search in one place; scales past what pgvector comfortably does at low ops cost; still light enough for a single workstation. |
| Local LLM runtime | **Ollama** (wraps llama.cpp) for MVP; `ILLMProvider` also implements a raw **llama.cpp server** backend and an **OpenAI-compatible** backend (vLLM/LM Studio/hosted) | Ollama = fastest path to a working 4-bit GGUF model with GPU offload control (`num_gpu` layers) and a stable HTTP API; llama.cpp server backend kept for when finer-grained control (batching, grammar-constrained decoding) is needed. |
| Embeddings | **BAAI/bge-small-en-v1.5** (or `bge-base` if VRAM allows) served via `sentence-transformers` on CPU or GPU | see §6 for shortlist/criteria. |
| Structured generation | **JSON Schema + Pydantic**, enforced via Ollama/llama.cpp **grammar-constrained decoding** where available, otherwise strict parse+retry | never trust free-form JSON from an LLM (§52). |
| Math verification | **SymPy** | deterministic algebra/calculus checking (§22). |
| Notation rendering | **MathJax 3** (client-side) + optional **mhchem** extension | LaTeX stored as text, never rasterized (§16–19). |
| Frontend | **React + TypeScript + Vite**, TanStack Query for API state, MathJax React wrapper for rendering | standard, no exotic dependency; kept decoupled from backend via REST only. |
| Auth | **OAuth2/OIDC-compatible** (e.g. Keycloak or the university's existing IdP) issuing JWTs consumed by FastAPI; local username/password fallback for MVP | universities usually already run an IdP (§34). |
| Containerization | **Docker Compose** for single-workstation deployment; Kubernetes manifests deferred to [FUTURE] multi-GPU/server scale-out | §49 explicitly asks to avoid unnecessary infra for an 8GB single-GPU box. |
| Observability | **Prometheus + Grafana** [PROD], structured JSON logging (`structlog`) from day one [MVP] | §43. |

Nothing here is vendor-locked: `ILLMProvider`, `IEmbeddingProvider`, and
`IVectorStore` are interfaces; swapping Ollama→vLLM or Qdrant→pgvector is a
config + adapter change, not a rewrite.

---

## 2. System Architecture

```
                                   ┌───────────────────────────┐
                                   │   Existing University DB   │
                                   │ (Program/Semester/Subject/ │
                                   │  Topic — MySQL/PG/other)   │
                                   └──────────────┬─────────────┘
                                                  │ read-only, via
                                                  │ AcademicDataPort adapter
                                                  ▼
┌───────────────┐   REST    ┌──────────────────────────────┐        ┌────────────────────┐
│   Admin/       │◄─────────►│         API Service          │        │   Object storage    │
│  Reviewer UI   │           │  (FastAPI, authn/z, audit)   │───────►│ (source PDFs, docs)  │
│  (React SPA)   │           └───────────────┬───────────────┘        └────────────────────┘
└───────────────┘                            │ enqueue
                                              ▼
                                   ┌────────────────────┐
                                   │   Redis (broker)    │
                                   └──────┬──────┬───────┘
                     ┌────────────────────┘      └────────────────────┐
                     ▼                                                 ▼
          ┌─────────────────────┐                          ┌─────────────────────┐
          │ Retrieval Workers    │                          │ Generation Workers   │
          │ (fetch/clean/chunk/  │                          │ (planner→LLM calls   │
          │  embed → Qdrant)     │                          │  →structured JSON)   │
          └──────────┬───────────┘                          └──────────┬───────────┘
                     │                                                 │
                     ▼                                                 ▼
          ┌─────────────────────┐                          ┌─────────────────────┐
          │      Qdrant          │◄────── retrieval ───────│ Validation Workers   │
          │ (chunk embeddings)   │        for generation &  │ (structural, answer, │
          └─────────────────────┘        verification       │  distractor, math/   │
                                                              │  chem, dedup, score) │
                                                              └──────────┬───────────┘
                                                                         │
                                                                         ▼
                                                              ┌─────────────────────┐
                                                              │   PostgreSQL         │
                                                              │ (jobs, candidates,   │
                                                              │  reviews, versions,  │
                                                              │  full provenance)    │
                                                              └─────────────────────┘
                                                                         ▲
                                              ┌──────────────────────────┘
                                              │
                                   ┌─────────────────────┐
                                   │  Local LLM Server    │  (Ollama / llama.cpp,
                                   │  — NEVER public       │   internal network only)
                                   └─────────────────────┘
```

Key architectural rule (§52): the local LLM endpoint is reachable only from
the worker network segment, never from the API service's public interface
and never from the Internet.

---

## 3. End-to-End Data Flow

1. Admin selects Program → Semester → Subject → Topics in the UI and defines
   a **Generation Job**: target approved count, option count, difficulty/Bloom
   distribution, source policy (local docs only / + approved Internet), model
   profile.
2. API validates the request against the `AcademicDataPort` (subject/topics
   must exist) and writes a `generation_jobs` row (+ `generation_job_topics`),
   status `QUEUED`.
3. **Knowledge preparation**: Retrieval workers pull/ingest configured local
   documents and (if enabled) run controlled web retrieval against
   `ApprovedDomain`s; text is cleaned, chunked, embedded, and stored in Qdrant
   with metadata (`subject_id, topic_id, source_id, chunk_id, page, section`).
4. **Topic Knowledge Pack** builder queries Qdrant per topic/subtopic and
   assembles a bounded, structured context object (definitions, formulas,
   misconceptions, cited chunks) — this, not raw web text, is what generation
   prompts see.
5. **Generation Planner** turns the requested total into a matrix of
   (topic × subtopic × Bloom level × difficulty × question type) cells with
   target counts, applying an **over-generation factor** (§30) derived from
   the currently observed rejection rate for that model/subject (default
   1.3–1.5x until real data exists).
6. **Generation workers** process the plan in small batches (10–30 questions
   per LLM call), each batch scoped to one knowledge pack slice, requesting
   strict JSON matching the MCQ schema (§12).
7. Each raw candidate is stored (`mcq_candidates`, status `GENERATED`) with
   full provenance (job, model, model version, prompt version, source chunk
   IDs) before any validation runs — provenance is never lost even for
   candidates that end up rejected.
8. **Validation pipeline** (separate workers, separate stage from generation,
   per §4/§52) runs, in order: structural (schema/option-count/length) →
   independent answer verification (second-pass LLM + SymPy where
   applicable) → distractor quality → math/chemistry notation validity →
   semantic deduplication → difficulty/Bloom classification/consistency →
   composite quality scoring. Each stage writes an `mcq_validation_results`
   row; status advances per the lifecycle in §26 or terminates into
   `REJECTED` / `DUPLICATE` / `INVALID` / `LOW_CONFIDENCE`.
9. Survivors reach `PENDING_REVIEW`. Generation continues until
   `approved-eligible candidate count >= target` or a configured max-attempt
   ceiling is hit, at which point the job is marked `COMPLETED` (possibly
   short of target, reported honestly, never silently topped up with
   low-quality filler).
10. **Human review**: reviewers work the `PENDING_REVIEW` queue; actions
    (approve/reject/edit/regenerate/flag/change difficulty/change Bloom) are
    fully versioned (`mcq_versions`) and audited (`mcq_reviews`).
11. `APPROVED` questions are the durable question bank, queryable via the
    export/exam-blueprint APIs (§40/§41), decoupled from any single exam.

Progress at every stage is polled via `GET /api/generation-jobs/{id}` (no
long-held HTTP requests — §29).

---

## 4. Database ERD

### 4.1 Academic master data (read via adapter — NOT owned by AI-QBE)

```
Program ──< Semester ──< Subject ──< Topic ──< SubTopic
```
AI-QBE stores only foreign *references* (`external_subject_id`,
`external_topic_id`, plus a denormalized label snapshot for display/audit
even if the source record changes later) — never a copy of the
university's authoritative rows.

### 4.2 AI-QBE owned schema (PostgreSQL)

```
generation_models
  id, name, provider_type, version, quantization, context_window,
  vram_estimate_mb, is_active

prompt_templates
  id, prompt_key, version, template_text, model_id (nullable = model-agnostic),
  temperature, top_p, top_k, repeat_penalty, max_tokens, created_at

generation_jobs
  id, external_subject_id, subject_label_snapshot, requested_count,
  option_count, difficulty_distribution (jsonb), bloom_distribution (jsonb),
  source_policy (jsonb), internet_enabled, model_id, status, created_by,
  created_at, started_at, completed_at, over_generation_factor,
  max_attempts, generated_count, approved_count, rejected_count,
  duplicate_count

generation_job_topics
  id, generation_job_id FK, external_topic_id, topic_label_snapshot,
  target_count, weight

academic_sources
  id, source_type (upload|url|instructor_provided), url, title, author,
  publisher, license, retrieval_date, approved_domain_id FK NULL,
  document_hash, storage_path

source_documents
  id, academic_source_id FK, mime_type, page_count, extracted_text_hash,
  ingestion_status

document_chunks
  id, source_document_id FK, external_subject_id, external_topic_id,
  chunk_index, page, section, text, embedding_vector_id (Qdrant point id),
  token_count

approved_domains / blocked_domains
  id, domain, priority, min_quality_score, notes

mcq_candidates
  id, generation_job_id FK, generation_job_topic_id FK, prompt_template_id FK,
  generation_model_id FK, model_version, batch_id, question_stem,
  bloom_level, difficulty, question_type, confidence, status,
  status_history (jsonb / see mcq_status_events), created_at

mcq_options
  id, mcq_candidate_id FK, option_index, text, is_correct, latex_flag

mcq_sources
  id, mcq_candidate_id FK, document_chunk_id FK, relevance_score

mcq_validation_results
  id, mcq_candidate_id FK, validator_name, stage, result (pass|fail|uncertain),
  score, details (jsonb), created_at

mcq_quality_scores
  id, mcq_candidate_id FK, factual_correctness, source_grounding,
  clarity, distractor_quality, single_correctness, topic_relevance,
  difficulty_match, bloom_match, option_conciseness, notation_validity,
  duplicate_risk, composite_score

mcq_reviews
  id, mcq_candidate_id FK, reviewer_id, action (approve|reject|edit|
  regenerate|flag|change_difficulty|change_bloom), comment, created_at

mcq_versions
  id, mcq_candidate_id FK, version_number, snapshot (jsonb), edited_by,
  created_at

generation_metrics
  id, generation_job_id FK, metric_name, metric_value, recorded_at

-- [FUTURE] analytics, populated post-deployment from real exam attempts
mcq_item_stats
  id, mcq_candidate_id FK, attempt_count, correct_count, incorrect_count,
  difficulty_index, discrimination_index, option_selection_distribution (jsonb)
```

Every `mcq_candidates` row is traceable end-to-end: job → topic → model →
model version → prompt version → source chunks (via `mcq_sources`) →
validation results → quality score → review history → versions. This
satisfies the "never lose provenance" rule (§52) by construction — provenance
columns are `NOT NULL` foreign keys, not optional metadata.

Indexes: composite `(generation_job_id, status)` on `mcq_candidates` for
queue queries; GIN index on `mcq_candidates.question_stem` (trigram) for
lexical dedup pre-filtering; standard FK indexes throughout. DDL lives in
[`schema.sql`](schema.sql).

---

## 5. Local LLM Candidates for an 8GB GPU

To be **empirically benchmarked in Phase 1** — this is a shortlist to
benchmark, not a final selection (§45 forbids picking by parameter count
alone, and §54 forbids claiming unmeasured numbers).

| Model | Params | Quant | Est. VRAM (4-bit) | Rationale to test |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (GGUF) | 7B | Q4_K_M | ~4.5–5.5GB | strong structured-output/JSON and math reasoning reputation; good multilingual if the university needs non-English. |
| Llama-3.1-8B-Instruct (GGUF) | 8B | Q4_K_M | ~5–6GB | widely validated instruction-following baseline. |
| Mistral-7B-Instruct-v0.3 (GGUF) | 7B | Q4_K_M | ~4.5GB | fast, historically decent JSON compliance; comparison baseline. |
| Gemma-2-9B-it (GGUF) | 9B | Q4_K_M | ~6–7GB | tight fit — included to see if quality gain justifies the VRAM headroom cost. |
| Phi-3.5-mini-instruct (GGUF) | 3.8B | Q4/Q8 | ~2.5–4GB | small/fast; candidate for the **verifier** role (cheap second pass) even if not the generator. |

All are run through the same benchmark harness (§45): tokens/sec, questions/
minute, JSON structural success %, validation pass %, VRAM/RAM footprint,
and the ~100-task academic-quality benchmark (§44) covering math/physics/
chemistry MCQ generation and answer verification. A **two-model split**
(one generator, one cheaper verifier) is the working hypothesis to evaluate,
since §21 requires an independent verification pass rather than trusting the
generator's own answer.

---

## 6. Recommended Embedding Model

Shortlist for CPU/GPU-light operation:

| Model | Dim | Notes |
|---|---|---|
| **BAAI/bge-small-en-v1.5** (initial default) | 384 | strong MTEB retrieval score for its size, runs comfortably on CPU, leaves full VRAM for the LLM. |
| BAAI/bge-base-en-v1.5 | 768 | better quality if GPU headroom exists after LLM load; evaluate in Phase 1/3. |
| intfloat/e5-small-v2 | 384 | alternative if bge licensing/behavior is unsuitable. |

Chunk-level embeddings are computed once at ingestion time and cached
(embedding is not on the generation hot path), so a slightly heavier model
is affordable even on modest hardware — final choice is decided empirically
in Phase 3 against retrieval-quality metrics (recall on a held-out set of
topic/question pairs), not by benchmark leaderboard alone.

---

## 7. Vector Database Decision

**Qdrant**, self-hosted (single Docker container for MVP), chosen over:

- **pgvector** — simplest ops (one less service) but weaker payload-filter +
  ANN performance at the scale implied by "millions of questions eventually"
  (§33), and mixes transactional and vector workloads in one DB. Kept as the
  documented fallback for installations that want to minimize service count;
  the `IVectorStore` interface makes this swap mechanical.
- **FAISS** — excellent ANN library but no built-in server, metadata
  filtering, or persistence story; would require building exactly what
  Qdrant already provides.
- **Milvus/Weaviate** — heavier operational footprint than justified for a
  single 8GB-GPU workstation deployment (§49 explicitly discourages
  unnecessary infrastructure).

Qdrant's payload filtering (`subject_id`, `topic_id`, `source_id`) combined
with vector similarity is exactly the "semantic relevance + metadata
filters" retrieval mode required by §7.

---

## 8. Internet Retrieval Architecture

```
Topic → Query Formulation → Search Adapter (pluggable: SearXNG self-hosted
  default; commercial API optional) → candidate URLs
    → ApprovedDomain / BlockedDomain filter
    → SourcePriority + MinimumSourceQuality gate
    → Fetch (sandboxed HTTP client, timeouts, size caps, robots.txt respect)
    → Content-type allowlist (html/pdf only)
    → Sanitization (strip scripts/styles/HTML, extract main content via
      readability-style extraction)
    → Prompt-injection scrub (see §9 below)
    → academic_sources + source_documents row, hash-deduped
    → into the same RAG ingestion pipeline as uploaded documents
```

Defaults: internet retrieval is **opt-in per generation job** and defaults
to **off**; `ApprovedDomain` starts with a small curated seed list (OER
repositories, university's own domain, standards bodies, e.g. NIST/IUPAC
for units/nomenclature) that an administrator extends explicitly — nothing
is auto-approved. Every fetched page is stored with retrieval date, URL,
title, author-if-available, publisher, and content hash for §36's
copyright/provenance governance.

## 9. Prompt-Injection & Untrusted-Content Defense

Retrieved text (web or uploaded documents) is **data, never instructions**,
enforced structurally, not just by prompt wording:

- Retrieved chunks are injected into prompts inside clearly delimited,
  labeled blocks (e.g. fenced with an unguessable-per-request boundary token)
  with an explicit system instruction that content inside the block is
  reference material only and any imperative sentences inside it must be
  ignored.
- A pre-filter strips common injection patterns (e.g. "ignore previous
  instructions", role-switch attempts, markdown/HTML that could be
  misrendered) before a chunk is ever embedded or shown to the LLM.
- Structured-output enforcement (JSON schema validation, §12) means even a
  successfully injected instruction cannot escape the response contract
  silently — a response that doesn't validate is discarded, not "repaired"
  by trusting more LLM output.
- The generation LLM has no tool-use / function-calling access to anything
  that could act on injected instructions (no filesystem, no network, no
  code execution) — it only ever returns text.

---

## 10. RAG Architecture

Document → clean (strip boilerplate/headers/footers/scripts) → normalize
(unicode, whitespace, encoding) → **semantic chunking** (target ~300–500
tokens per chunk, sentence/heading-boundary aware rather than fixed-width,
to keep formulas and their explanations together) → embed (bge-small) →
upsert into Qdrant with full metadata → retrieval at generation/verification
time uses `topic_id`/`subject_id` filter + top-k cosine similarity, k
configurable (default 6–10), then deduplicated/merged into the Topic
Knowledge Pack (§ per master prompt §8) which is capped to a bounded token
budget appropriate to the model's context window (e.g. ≤2–3k tokens of
context per generation batch, leaving room for the instruction + few-shot
schema example + output budget) — this is the "bounded context windows"
requirement for 8GB-GPU feasibility.

---

## 11. MCQ Generation Architecture

- **Planner** (§9 of the master prompt): converts `requested_count` into a
  weighted matrix over topic × subtopic × Bloom × difficulty × question
  type, informed by `weight`/teaching-hours fields on
  `generation_job_topics` when supplied, else uniform.
- **Batch executor**: each Celery task handles one plan cell, 10–30 items,
  building the prompt from `prompt_templates` (versioned) + the relevant
  Topic Knowledge Pack slice, calling `ILLMProvider.generate_structured()`
  which enforces the JSON schema (grammar-constrained decoding when the
  backend supports it, else strict-parse-with-bounded-retries).
- **Over-generation controller**: tracks rolling accept-rate per
  (subject, model, prompt_version) and adjusts how many additional batches
  to schedule so the job converges on the approved target without a fixed
  guess (§30).
- Generation and validation are **separate Celery task chains** — a
  generation worker never runs the validation logic itself (§4 requirement:
  generation and validation must be separate stages), which also lets
  validation be re-run independently (e.g. after a validator bugfix) without
  re-generating.

---

## 12. Validation Architecture

Ordered pipeline, each a separate, independently testable module, each
writing to `mcq_validation_results`:

1. **Structural validator** — JSON-schema conformance, exact option count,
   exactly one `is_correct`, option word-count bounds (1–6 words, flag >6),
   no empty fields.
2. **Answer verifier** — independent pass: (a) SymPy deterministic check for
   symbolic/numeric math items; (b) second LLM pass (different prompt,
   ideally the smaller "verifier" model from §5) given the same evidence,
   asked to independently derive the answer and compare; (c) consensus rule:
   PASS only if deterministic-or-verifier agrees with the generator;
   disagreement → `UNCERTAIN` → forced human review, never auto-rejected
   *or* auto-approved.
3. **Distractor validator** — rule-based + embedding-similarity checks for:
   duplicate options, semantically-equivalent options, grammatical
   agreement clues (e.g. "an" before a consonant-starting option), abnormal
   length outliers, off-domain/absurd distractors (embedding distance from
   the question's concept cluster beyond a threshold), multiple-correct
   detection (each option re-scored against the evidence independently).
4. **Math/Physics/Chemistry notation validator** — LaTeX parse-ability
   (balanced braces/environments, MathJax-supported command allowlist),
   unit-presence/plausibility checks for physics numerics, chemical formula
   parseability (subscript/charge balance) and, if `mhchem` enabled,
   `\ce{}` syntax validation.
5. **Semantic deduplication** — 4-level per §23: exact normalized-hash →
   lexical (trigram/Levenshtein) → embedding cosine similarity against
   already-accepted items in the same topic → LLM-judge tiebreak only for
   borderline similarity scores (configurable band, e.g. 0.80–0.92 cosine),
   to avoid burning LLM calls on every pair.
6. **Difficulty/Bloom classifier** — a checklist-scored (not purely
   LLM-vibes) estimate per §11: number of reasoning steps, presence/absence
   of calculation, distractor-similarity-derived difficulty, cross-checked
   against the LLM's self-reported label; mismatch beyond a threshold flags
   for review rather than silently overriding.
7. **Quality scoring** — composite of the eleven dimensions in §25,
   *computed from the above validators' outputs*, never from LLM
   self-confidence alone (§52); weights configurable, stored per candidate
   in `mcq_quality_scores`.

Status transitions strictly follow §26's lifecycle; any validator producing
`FAIL` moves the candidate to a terminal non-approved status with the
specific reason preserved (not just "rejected").

---

## 13. Mathematics / Physics / Chemistry Strategy

- All notation authored/stored as **LaTeX text** (MathJax-compatible),
  never rasterized to images (§16).
- Chemistry: **mhchem** support is **recommended but configurable** — decide
  per-deployment based on whether the reviewing faculty already expect
  `\ce{}` syntax; the notation validator supports both mhchem and plain
  LaTeX subscript/superscript chemical notation so this is a config flag,
  not an architectural fork.
- Physics: units are validated as a **distinct check** from LaTeX syntax
  (a dimensionally-plausible but syntactically valid expression can still be
  wrong) — a lightweight unit-consistency checker (regex/lookup against a
  small SI-unit table) flags missing or inconsistent units in numeric
  answers/options.
- Math answer correctness: wherever the item is expressible as a symbolic
  or numeric computation, SymPy independently evaluates it; the LLM's stated
  correct option must match the computed value within a defined numeric
  tolerance — this is the primary "don't trust LLM reasoning for computable
  answers" control (§22).

---

## 14. Security Architecture

- **AuthN**: OIDC/JWT (or local fallback for MVP dev); no anonymous API
  access beyond a health check.
- **AuthZ**: RBAC with the six roles from §34 (Administrator,
  QuestionGenerator, AcademicReviewer, SubjectExpert, ExamController,
  ReadOnly), enforced at the FastAPI dependency-injection layer per route.
- **Network isolation**: LLM server and vector store live on an internal
  Docker network with no published ports beyond the API service; only the
  API and worker containers can reach them.
- **Input validation**: Pydantic models on every request; upload size/type
  caps on document ingestion; URL allowlisting for internet retrieval (§8).
- **Secrets**: environment-injected via Docker secrets / `.env` (never
  committed); rotated independently of application deploys.
- **Rate limiting**: per-user/per-role limits on job-creation and review
  endpoints (e.g. `slowapi`/reverse-proxy level) to bound worst-case load on
  the single shared GPU.
- **Audit log**: every state-changing action (job create/start/pause, review
  action, approval) appended to an immutable audit table with actor, IP,
  timestamp, before/after snapshot — separate from `mcq_reviews` (which is
  the academic record) so security audit isn't lost even if academic tables
  are pruned.

---

## 15. Proposed REST API (representative, not exhaustive)

```
GET    /api/subjects
GET    /api/subjects/{id}/topics
POST   /api/generation-jobs
GET    /api/generation-jobs
GET    /api/generation-jobs/{id}
GET    /api/generation-jobs/{id}/progress
POST   /api/generation-jobs/{id}/start
POST   /api/generation-jobs/{id}/pause
POST   /api/generation-jobs/{id}/resume
POST   /api/generation-jobs/{id}/cancel

GET    /api/questions                 (filters: subject, topic, status,
                                        difficulty, bloom, job_id)
GET    /api/questions/{id}
GET    /api/questions/{id}/versions
GET    /api/questions/{id}/evidence
POST   /api/questions/{id}/approve
POST   /api/questions/{id}/reject
POST   /api/questions/{id}/edit
POST   /api/questions/{id}/regenerate
POST   /api/questions/{id}/flag
POST   /api/questions/bulk-review

POST   /api/sources                   (upload / register URL)
GET    /api/sources/{id}

POST   /api/exam-blueprints
GET    /api/exam-blueprints/{id}/preview
POST   /api/exam-blueprints/{id}/select   (returns randomized subset)

GET    /api/export/{job_id}?format=json|csv|xlsx

GET    /api/models                    (registered generation_models)
GET    /api/prompts                   (versioned prompt_templates, read-only
                                        via API; edited via migration/admin
                                        tooling, not free-text API PUT)
```

All list endpoints are paginated; all mutating endpoints require RBAC and
emit an audit record.

---

## 16. Project Directory Structure

```
ai-qbe/
├── backend/
│   ├── api/            # FastAPI routers, request/response schemas
│   ├── domain/         # entities, enums, business rules (status lifecycle)
│   ├── database/       # SQLAlchemy models, Alembic migrations, AcademicDataPort
│   ├── generation/      # planner, batch executor, over-generation controller
│   ├── validation/      # structural, answer, distractor, notation, dedup, scoring
│   ├── retrieval/       # web search adapter, domain policy, fetch/sanitize
│   ├── embeddings/      # embedding provider interface + implementations
│   ├── models/          # ILLMProvider + provider implementations
│   ├── workers/         # Celery tasks/chains
│   └── security/        # auth, RBAC, audit logging
├── frontend/            # React + TS SPA
├── llm/
│   ├── providers/       # Ollama / llama.cpp / OpenAI-compatible adapters
│   ├── prompts/         # versioned template files
│   └── schemas/         # JSON Schemas for MCQ/knowledge-pack objects
├── rag/                 # chunking, ingestion pipeline, Qdrant client wrapper
├── tests/               # unit/integration/API/schema/RAG/dedup/math/chem
├── scripts/             # benchmark harness, seed data, migration helpers
├── deployment/          # docker-compose.yml, Dockerfiles, env templates
├── docs/                # this design doc, ERD, runbooks
└── README.md
```

---

## 17. Deployment Architecture

**MVP / single-workstation (8GB GPU, 32GB RAM) — Docker Compose:**

```yaml
services: api, worker, postgres, redis, qdrant, ollama, frontend
```

`ollama` and `qdrant` are the GPU/RAM-heavy services; `worker` concurrency is
capped (e.g. 1–2 generation workers) to avoid oversubscribing the single
GPU — the LLM provider abstraction serializes/queues GPU-bound calls
regardless of worker count so this is a safe default, not a hard limit.

**Native deployment alternative** is documented for institutions preferring
no Docker: systemd units per service, a native Postgres/Redis install, and
Ollama run as a system service — same codebase, no compose dependency.

**Scale-out [FUTURE]**: when moving to a larger GPU/server, only the
`ollama`/LLM service and worker replica count change (e.g. swap to a vLLM
OpenAI-compatible backend on a bigger GPU); API, DB, and vector-store
architecture are unchanged, which is the point of the `ILLMProvider`
abstraction.

---

## 18. Expected Performance Constraints (targets, not measurements)

These are **planning assumptions to validate in Phase 1/9**, explicitly not
claimed as measured (§54):

- Single 8GB GPU realistically runs **one** 7–9B Q4 model with useful
  context (2–4k tokens) at a time; concurrent generation + verification
  passes will contend for the same GPU unless the verifier is a much
  smaller/CPU-viable model — this is exactly why §5 evaluates a
  generator/verifier split.
- Expect generation throughput in the low tens of validated-candidate
  batches per hour on this hardware, not thousands of questions in minutes;
  the over-generation + async-job design (§29–30) exists specifically
  because a 3,000-question job is expected to run for an extended background
  period, not a single request lifetime.
- RAG ingestion (chunk+embed) is comparatively cheap and CPU-viable, so it
  should not be the bottleneck; the LLM generation/verification calls are.

Phase 1 and Phase 9 replace every number above with measured data.

---

## 19. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 8GB VRAM insufficient for acceptable generation quality at usable speed | Phase 1 benchmark compares multiple quantized models before committing; fallback to smaller/faster model + heavier reliance on deterministic validation (SymPy, rule-based distractor checks) to compensate for a weaker generator. |
| LLM hallucinated/unsupported facts entering the bank | Mandatory source-grounding requirement (§20) — no evidence, no question — plus independent answer verification (§21) before anything reaches `PENDING_REVIEW`. |
| Malformed JSON stalls generation throughput | Grammar-constrained decoding where supported; bounded retry with backoff; malformed-after-retries candidates are logged and skipped, not silently dropped (tracked in `generation_metrics`). |
| Duplicate/low-diversity question banks at scale | 4-level dedup (§23) + concept-coverage diversity metrics (§24) enforced before `APPROVED`, not just at export time. |
| Prompt injection via retrieved web content | Structural delimiting + pre-filtering + no tool access from the generation LLM (§9). |
| Copyright exposure from scraped textbook content | `ApprovedDomain`/license metadata capture (§36) + instruction to synthesize rather than quote; large verbatim overlap is itself a dedup/plagiarism check candidate for Phase 4. |
| Answer-position bias | Deterministic post-validation randomization pass that preserves the correct-answer mapping (§15), audited via a per-job position-distribution metric. |
| Long-running jobs lost on crash/restart | Celery task idempotency + checkpointed job state in Postgres (§46) — jobs resume from last completed batch, not from zero. |
| Existing university DB schema differs from assumptions | `AcademicDataPort` interface isolates all schema-specific logic to one adapter module per institution; core app never queries university tables directly. |

---

## 20. Implementation Roadmap & Phase Acceptance Criteria

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| **0** (this doc) | Architecture, ERD, stack, roadmap | Reviewed and approved before Phase 1 starts. |
| **1** | Local LLM benchmark harness + report | ≥3 quantized models benchmarked on the target 8GB GPU across the ~100-task set; a generation model and (candidate) verifier model selected **with measured numbers**, not assumptions. |
| **2** | Core DB + API, manual-context generation | Subjects/topics readable via adapter; generation job CRUD + status lifecycle working; MCQs generated from manually-supplied context (no RAG yet); full provenance chain populated end-to-end; API tests passing. |
| **3** | Document RAG | Upload → clean → chunk → embed → Qdrant → topic knowledge pack retrieval demonstrated against real instructor-provided material, with citations traceable to source chunk/page. |
| **4** | Controlled Internet retrieval | Approved-domain-gated retrieval merges into the same knowledge pack; prompt-injection test suite (malicious/instructional web content) passes without behavior change. |
| **5** | MCQ generation engine | Planner correctly distributes a requested total across topic/Bloom/difficulty cells; batches generate schema-valid JSON at a measured success rate; option-count/length constraints enforced. |
| **6** | Scientific content support | Math/physics/chemistry sample set renders correctly in MathJax; notation validator rejects a seeded set of malformed LaTeX/chemical-formula test cases with zero false negatives on that set. |
| **7** | QA pipeline | All validators (structural, answer, distractor, notation, dedup, scoring) implemented and unit-tested; no candidate reaches `APPROVED` without passing every stage; quality score demonstrably derived from validator outputs (traceable, not opaque). |
| **8** | Human review system | Reviewer can approve/reject/edit/regenerate with MathJax-rendered evidence panel; edits versioned; bulk review functional. |
| **9** | Large-scale generation test | Successful, measured runs at 100/500/1,000/3,000 requested questions with recorded time, rejection rate, duplicate rate, approved yield, GPU/RAM utilization — used to tune batching/concurrency. |
| **10** | Exam integration | Blueprint-driven selection meets topic/difficulty/Bloom distribution constraints with no duplicate-concept violations; exports validated against schema. |
| **11** | Production hardening | Backups verified restorable; monitoring dashboards live; job-resume tested against simulated worker/LLM crash; security review complete; deployment docs validated by a clean-environment install. |

---

## 21. Open Questions for Sign-off

1. Which existing university DB dialect/schema should the first
   `AcademicDataPort` adapter target (MySQL/MariaDB assumed above — confirm)?
2. Is `mhchem` chemistry notation the faculty's expectation, or is plain
   LaTeX subscript/superscript sufficient for MVP?
3. Does the university have an existing OIDC/IdP to integrate with, or should
   MVP auth be local-only initially?
4. Any pre-approved OER/textbook sources to seed `ApprovedDomain`s with, or
   should Phase 4 start from an empty allowlist?

Once these are answered (or explicitly deferred with a documented default),
Phase 1 (local LLM benchmarking) begins.
