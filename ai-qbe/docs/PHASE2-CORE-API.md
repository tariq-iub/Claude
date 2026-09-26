# AI-QBE — Phase 2: Core Database & API

Status: **Implemented and tested (54/54 tests passing).** No RAG, no
Internet retrieval, no independent fact-verification pipeline yet — all
correctly out of scope until Phases 3, 4, and 7 respectively.

## 1. What was implemented

- **`backend/database/models.py`** — SQLAlchemy 2.0 models mirroring
  `docs/schema.sql` exactly (same tables, columns, constraints), dialect-
  agnostic (Postgres in production, SQLite for tests — a custom
  `JSONVariant` type decorator gives JSONB on Postgres and JSON-on-TEXT on
  SQLite from the same model field). Includes the RAG-related tables
  (`AcademicSource`, `SourceDocument`, `DocumentChunk`, `MCQSource`) even
  though Phase 2 doesn't populate them, so Phase 3 adds behavior, not
  schema; plus a `User` table for the local-auth fallback and a separate
  `AuditLogEntry` table for the security audit trail (kept apart from
  `MCQReview`, the academic record, per section 14 of the Phase 0 design).
- **`backend/database/academic_port.py`** — the `AcademicDataPort`
  interface (list/get subjects, list/get topics) plus
  `InMemoryAcademicDataPort`, a demo adapter seeded with the Physics-I
  example from the master prompt's business-objective section. No code
  outside this module ever queries "university data" directly — swapping
  in a real institution's MySQL/MariaDB schema is one new adapter class.
- **`backend/database/migrations/`** — Alembic, configured to read
  `AIQBE_DATABASE_URL` and to autogenerate against the SQLAlchemy models.
  The initial migration (`versions/a3b78f5bbd0b_initial_schema.py`) was
  generated and verified to `upgrade head` / `downgrade base` cleanly
  against a real SQLite database (see `tests/test_alembic_migration.py`).
- **`backend/domain/enums.py`** — `GenerationJobStatus`, `MCQStatus`,
  `BloomLevel`, `DifficultyLevel`, `ReviewAction`, `Role`, plus the RBAC
  access-matrix sets, all as the single source of truth for status
  strings shared by the DB, API, and business logic.
- **`backend/generation/planner.py`** — largest-remainder apportionment of
  a job's `requested_count` across topics (by weight) and, within each
  topic, across the job's difficulty/Bloom distributions — guaranteed to
  sum exactly to what was requested, not an approximation that can drift.
- **`backend/generation/executor.py`** — runs a job's plan against a
  configured `ILLMProvider` using **manually-supplied per-topic context**
  (the Phase 2 "no RAG yet" requirement), persists every candidate with
  full provenance before any validation decision, and applies structural
  validation immediately.
- **`backend/validation/structural.py`** — the first Phase 7 validation
  stage, pulled forward: JSON Schema conformance, option-count/word-count
  limits, duplicate-option detection, correct-option range checking. A
  candidate that fails this is marked `INVALID`, never written as if
  reviewable.
- **`backend/security/`** — local username/password auth issuing JWTs
  (`auth.py`, using `bcrypt` directly — see limitations for why not
  passlib), and centralized RBAC dependencies (`rbac.py`) implementing the
  six roles and access matrix from Phase 0 section 14. `audit.py` records
  every state-changing action to `AuditLogEntry`.
- **`backend/api/`** — FastAPI app (`main.py`) with routers for `/api/auth`,
  `/api/subjects`, `/api/generation-jobs` (create/list/get/progress/
  start/pause/resume/cancel), and `/api/questions` (list/get/approve/
  reject/edit/flag/regenerate), matching the REST surface proposed in
  `docs/PHASE0-DESIGN.md` section 15.
- **`backend/workers/`** — a Celery app + a `run_generation_job_task` that
  wraps the same executor function production code will call. Runs in
  **eager mode** (`AIQBE_CELERY_TASK_ALWAYS_EAGER=true`) by default so the
  API and test suite work without a Redis broker or a separate worker
  process — see limitations for exactly what that trades away.
- **`llm/providers/mock_provider.py`** — a deterministic, non-LLM
  `MockProvider` used as the default in dev/test, since this environment
  still has no GPU/LLM runtime (same constraint as Phase 1). Clearly
  documented as a development convenience, never a benchmarked or
  deployable model.
- **Tests** (`tests/test_planner.py`, `test_structural_validation.py`,
  `test_generation_executor.py`, `test_api_integration.py`,
  `test_alembic_migration.py`) — 54 tests total (including Phase 1's 36),
  all passing, covering: apportionment math, structural validation edge
  cases, end-to-end job execution against both a well-behaved and a
  broken provider, a full API integration flow (login → browse subjects/
  topics → create job → reject an unknown subject → start job → progress
  → list questions → RBAC-blocked approve → approve → reject with comment
  → edit with version history → job counters), and a real Alembic
  upgrade/downgrade round-trip.

```
$ python3 -m pytest tests/ -q
......................................................
54 passed in 5.58s
```

## 2. Design decisions

- **Provenance is written before any validation verdict.** Every
  `MCQCandidate` row is created (with job/topic/model/prompt-template
  foreign keys populated) *before* structural validation runs, so even an
  `INVALID` candidate is fully traceable — nothing is silently discarded
  pre-persistence.
- **Generation and validation stay separate function calls even without a
  queue between them yet.** `run_generation_job` calls the provider, then
  hands the raw parsed response to `validate_mcq_structure` as a distinct
  step — this is what lets Phase 7 slot in fact-verification/dedup/quality
  scoring as additional stages later without restructuring the executor.
- **Every generated candidate lands at `PENDING_REVIEW`, never
  `APPROVED`, in Phase 2.** There is no independent fact-verification yet,
  so nothing is approved automatically — this matches the master prompt's
  "never automatically trust an LLM-generated correct answer" rule
  precisely because Phase 2 has no mechanism to check it computationally
  or via a second model pass yet (that's Phase 7).
- **Bulk-review, edit-then-approve-in-one-call, and a real diff view are
  intentionally NOT built here.** Phase 8 owns the human-review UI/UX;
  Phase 2 only needs the underlying approve/reject/edit/regenerate
  primitives with correct versioning and audit, which the API now has.
- **`bcrypt` directly instead of `passlib[bcrypt]`.** `passlib` 1.7.4's
  bcrypt backend self-test raises against `bcrypt>=4.1` (a known,
  currently-unpatched upstream incompatibility) — hit and confirmed while
  building this phase. Calling the `bcrypt` library directly avoids a
  fragile dependency chain for something this small; if the project later
  wants passlib's multi-scheme flexibility, swap it back once upstream
  fixes the incompatibility.
- **Celery eager mode by default.** Per Phase 0's "operational simplicity"
  goal for a single-workstation deployment, requiring a live Redis broker
  and a separate worker process just to run the test suite or a local dev
  server would add friction with no corresponding benefit at this stage.
  `AIQBE_CELERY_TASK_ALWAYS_EAGER=false` plus a running
  `celery -A backend.workers.celery_app worker` is how a real deployment
  runs it — same task code, no changes needed.
- **Alembic env.py prefers an explicit programmatic `sqlalchemy.url` over
  the app's configured default**, so both CLI usage (`alembic upgrade
  head`, driven by `AIQBE_DATABASE_URL`) and the test suite (which points
  at a fresh scratch SQLite file per test) work without one clobbering
  the other.

## 3. Configuration

Environment variables (see `backend/config.py`), all prefixed `AIQBE_`:

| Variable | Default | Purpose |
|---|---|---|
| `AIQBE_DATABASE_URL` | `sqlite:///./aiqbe_dev.db` | SQLAlchemy URL; Postgres in production. |
| `AIQBE_JWT_SECRET_KEY` | dev-only placeholder | **Must** be overridden in any real deployment. |
| `AIQBE_JWT_EXPIRE_MINUTES` | 480 | Access token lifetime. |
| `AIQBE_CELERY_BROKER_URL` / `AIQBE_CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` / `/1` | Only used when eager mode is off. |
| `AIQBE_CELERY_TASK_ALWAYS_EAGER` | `true` | See design decisions above. |
| `AIQBE_DEFAULT_LLM_PROVIDER_TYPE` | `mock` | Set to `ollama`/`llamacpp`/`openai_compatible` once Phase 1 selects a real model. |

Running locally:

```bash
cd ai-qbe
pip install -r backend/requirements.txt -r llm/../scripts/benchmark/requirements.txt
export AIQBE_DATABASE_URL=sqlite:///./aiqbe_dev.db
cd backend && alembic upgrade head && cd ..
uvicorn backend.api.main:app --reload
# default seeded admin: admin / changeme123 -- rotate immediately
```

Running tests: `python3 -m pytest tests/ -q` from `ai-qbe/`.

## 4. Limitations

- **No real LLM was used** — `MockProvider` stands in until Phase 1's
  hardware benchmark (still pending real hardware, per
  `docs/PHASE1-BENCHMARK.md`) selects an actual model; swapping it in is a
  `generation_models`/job-config change, not a code change.
- **No RAG, no Internet retrieval.** Context is supplied manually per
  topic on job creation, exactly as Phase 2 specifies. `manual_context`
  is a plain string field on `GenerationJobTopic`; Phase 3 adds the
  chunk-retrieval path that populates the same downstream prompt slot.
- **No independent fact verification, deduplication, or quality
  scoring.** Structural validation only. This is why every valid
  candidate stops at `PENDING_REVIEW` rather than reaching `APPROVED`
  automatically — see design decisions above.
- **Celery eager mode means there is no real "RUNNING" window to
  observe** for progress polling yet — a job goes QUEUED → COMPLETED in
  one synchronous call. Pause/resume are implemented and status-guarded
  correctly, but won't do anything meaningful mid-job until Phase 5's
  batched execution exists.
- **Regeneration is single-item and synchronous**, calling the provider
  inline within the API request. Phase 5's batch executor will route this
  through the same queue as bulk generation instead.
- **Local username/password auth is an MVP fallback.** Institutional
  OIDC/IdP integration (Phase 0 open question #3) is not implemented; the
  `Role` string contract in `backend/security/rbac.py` is what an OIDC
  adapter would need to populate, so the switch doesn't touch route code.
- **The default seeded admin account** (`admin` / `changeme123`) is a dev
  convenience and must be rotated or disabled before any non-local
  deployment.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "Subjects/topics readable via adapter; generation job CRUD + status
> lifecycle working; MCQs generated from manually-supplied context (no RAG
> yet); full provenance chain populated end-to-end; API tests passing."

**Met**, within Phase 2's explicit scope (structural validation only, no
fact-checking/dedup/quality yet, no real LLM — see limitations above for
what's deliberately still missing before Phase 7 and a real model
selection close those gaps).
