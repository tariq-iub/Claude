-- AI-QBE Phase 0 — PostgreSQL DDL for the AI-QBE owned metadata schema.
--
-- This schema does NOT include the university's academic master data
-- (Program/Semester/Subject/Topic/SubTopic). Those are read through an
-- AcademicDataPort adapter against the existing institutional database and
-- are referenced here only via external_*_id + a denormalized label
-- snapshot, so this schema never assumes ownership of that data.
--
-- Design goals reflected below:
--   * every generated question is traceable end-to-end (job, topic, model,
--     model version, prompt version, source chunks, validation results,
--     quality score, review history, versions) via NOT NULL foreign keys
--     wherever the relationship is mandatory;
--   * generation and validation are modeled as separate stages/tables;
--   * status lifecycles are enums, not free text;
--   * indexes support the query patterns the API/worker layer needs
--     (job queues, dedup pre-filtering, review queues).

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------
-- Reference / configuration tables
-- ---------------------------------------------------------------------

CREATE TABLE generation_models (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    provider_type   TEXT NOT NULL,          -- 'ollama' | 'llamacpp' | 'openai_compatible'
    version         TEXT NOT NULL,
    quantization    TEXT,
    context_window  INTEGER NOT NULL,
    vram_estimate_mb INTEGER,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version, quantization)
);

CREATE TABLE prompt_templates (
    id              BIGSERIAL PRIMARY KEY,
    prompt_key      TEXT NOT NULL,           -- e.g. 'mcq_generation'
    version         INTEGER NOT NULL,
    template_text   TEXT NOT NULL,
    model_id        BIGINT REFERENCES generation_models(id),
    temperature     NUMERIC(3,2) NOT NULL DEFAULT 0.3,
    top_p           NUMERIC(3,2) NOT NULL DEFAULT 0.9,
    top_k           INTEGER,
    repeat_penalty  NUMERIC(3,2),
    max_tokens      INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (prompt_key, version)
);

CREATE TABLE approved_domains (
    id                  BIGSERIAL PRIMARY KEY,
    domain              TEXT NOT NULL UNIQUE,
    priority            INTEGER NOT NULL DEFAULT 100,
    min_quality_score   NUMERIC(5,2) NOT NULL DEFAULT 0,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE blocked_domains (
    id          BIGSERIAL PRIMARY KEY,
    domain      TEXT NOT NULL UNIQUE,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Generation jobs
-- ---------------------------------------------------------------------

CREATE TYPE generation_job_status AS ENUM (
    'QUEUED', 'PREPARING_KNOWLEDGE', 'RUNNING', 'PAUSED',
    'COMPLETED', 'FAILED', 'CANCELLED'
);

CREATE TABLE generation_jobs (
    id                      BIGSERIAL PRIMARY KEY,
    external_subject_id     TEXT NOT NULL,
    subject_label_snapshot  TEXT NOT NULL,
    requested_count         INTEGER NOT NULL CHECK (requested_count > 0),
    option_count            SMALLINT NOT NULL DEFAULT 4 CHECK (option_count BETWEEN 3 AND 5),
    difficulty_distribution JSONB NOT NULL,   -- {"easy":0.3,"medium":0.5,"hard":0.2}
    bloom_distribution      JSONB NOT NULL,   -- {"remember":0.25,...}
    source_policy           JSONB NOT NULL,   -- {"local_docs":true,"internet":false,...}
    internet_enabled        BOOLEAN NOT NULL DEFAULT FALSE,
    model_id                BIGINT NOT NULL REFERENCES generation_models(id),
    status                  generation_job_status NOT NULL DEFAULT 'QUEUED',
    created_by              TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    over_generation_factor  NUMERIC(4,2) NOT NULL DEFAULT 1.4,
    max_attempts            INTEGER NOT NULL DEFAULT 3,
    generated_count         INTEGER NOT NULL DEFAULT 0,
    approved_count          INTEGER NOT NULL DEFAULT 0,
    rejected_count          INTEGER NOT NULL DEFAULT 0,
    duplicate_count         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX ix_generation_jobs_status ON generation_jobs(status);

CREATE TABLE generation_job_topics (
    id                  BIGSERIAL PRIMARY KEY,
    generation_job_id   BIGINT NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    external_topic_id   TEXT NOT NULL,
    topic_label_snapshot TEXT NOT NULL,
    target_count        INTEGER NOT NULL CHECK (target_count >= 0),
    weight               NUMERIC(6,3) NOT NULL DEFAULT 1.0
);

CREATE INDEX ix_generation_job_topics_job ON generation_job_topics(generation_job_id);

-- ---------------------------------------------------------------------
-- Sources / RAG
-- ---------------------------------------------------------------------

CREATE TYPE source_type AS ENUM ('upload', 'url', 'instructor_provided');

CREATE TABLE academic_sources (
    id                  BIGSERIAL PRIMARY KEY,
    source_type         source_type NOT NULL,
    url                 TEXT,
    title               TEXT,
    author              TEXT,
    publisher           TEXT,
    license             TEXT,
    retrieval_date      TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_domain_id  BIGINT REFERENCES approved_domains(id),
    document_hash       TEXT NOT NULL,
    storage_path        TEXT,
    UNIQUE (document_hash)
);

CREATE TABLE source_documents (
    id                      BIGSERIAL PRIMARY KEY,
    academic_source_id      BIGINT NOT NULL REFERENCES academic_sources(id) ON DELETE CASCADE,
    mime_type               TEXT NOT NULL,
    page_count              INTEGER,
    extracted_text_hash     TEXT,
    ingestion_status        TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE document_chunks (
    id                      BIGSERIAL PRIMARY KEY,
    source_document_id      BIGINT NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    external_subject_id     TEXT NOT NULL,
    external_topic_id       TEXT,
    chunk_index             INTEGER NOT NULL,
    page                    INTEGER,
    section                 TEXT,
    text                    TEXT NOT NULL,
    embedding_vector_id     TEXT NOT NULL,   -- Qdrant point id (vectors live in Qdrant, not Postgres)
    token_count             INTEGER NOT NULL
);

CREATE INDEX ix_document_chunks_topic ON document_chunks(external_subject_id, external_topic_id);

-- ---------------------------------------------------------------------
-- MCQ candidates and their lifecycle
-- ---------------------------------------------------------------------

CREATE TYPE mcq_status AS ENUM (
    'GENERATED', 'STRUCTURE_VALIDATED', 'FACT_VALIDATED', 'DEDUPLICATED',
    'QUALITY_CHECKED', 'PENDING_REVIEW', 'APPROVED',
    'REJECTED', 'NEEDS_REVISION', 'DUPLICATE', 'INVALID', 'LOW_CONFIDENCE'
);

CREATE TYPE bloom_level AS ENUM ('remember', 'understand', 'apply', 'analyze');
CREATE TYPE difficulty_level AS ENUM ('easy', 'medium', 'hard');

CREATE TABLE mcq_candidates (
    id                          BIGSERIAL PRIMARY KEY,
    generation_job_id           BIGINT NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    generation_job_topic_id     BIGINT NOT NULL REFERENCES generation_job_topics(id),
    prompt_template_id          BIGINT NOT NULL REFERENCES prompt_templates(id),
    generation_model_id         BIGINT NOT NULL REFERENCES generation_models(id),
    model_version               TEXT NOT NULL,
    batch_id                    UUID NOT NULL,
    question_stem               TEXT NOT NULL,
    explanation                 TEXT,
    bloom_level                 bloom_level NOT NULL,
    difficulty                  difficulty_level NOT NULL,
    question_type                TEXT NOT NULL DEFAULT 'single_best_answer',
    confidence                  NUMERIC(4,3),
    status                      mcq_status NOT NULL DEFAULT 'GENERATED',
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_mcq_candidates_job_status ON mcq_candidates(generation_job_id, status);
CREATE INDEX ix_mcq_candidates_stem_trgm ON mcq_candidates USING gin (question_stem gin_trgm_ops);

CREATE TABLE mcq_options (
    id                  BIGSERIAL PRIMARY KEY,
    mcq_candidate_id    BIGINT NOT NULL REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    option_index        SMALLINT NOT NULL,      -- 0=A, 1=B, ...
    text                TEXT NOT NULL,
    is_correct          BOOLEAN NOT NULL DEFAULT FALSE,
    latex_flag          BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (mcq_candidate_id, option_index)
);

-- Enforce exactly one correct option per candidate for standard question types.
CREATE UNIQUE INDEX ux_mcq_options_single_correct
    ON mcq_options(mcq_candidate_id)
    WHERE is_correct;

CREATE TABLE mcq_sources (
    id                  BIGSERIAL PRIMARY KEY,
    mcq_candidate_id    BIGINT NOT NULL REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    document_chunk_id   BIGINT NOT NULL REFERENCES document_chunks(id),
    relevance_score     NUMERIC(5,4)
);

CREATE INDEX ix_mcq_sources_candidate ON mcq_sources(mcq_candidate_id);

CREATE TABLE mcq_validation_results (
    id                  BIGSERIAL PRIMARY KEY,
    mcq_candidate_id    BIGINT NOT NULL REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    validator_name      TEXT NOT NULL,   -- 'structural' | 'answer_verifier' | 'distractor' |
                                          -- 'notation' | 'dedup' | 'bloom_difficulty'
    stage               TEXT NOT NULL,
    result              TEXT NOT NULL CHECK (result IN ('pass', 'fail', 'uncertain')),
    score               NUMERIC(6,3),
    details             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_mcq_validation_results_candidate ON mcq_validation_results(mcq_candidate_id);

CREATE TABLE mcq_quality_scores (
    id                      BIGSERIAL PRIMARY KEY,
    mcq_candidate_id        BIGINT NOT NULL UNIQUE REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    factual_correctness     NUMERIC(5,2),
    source_grounding        NUMERIC(5,2),
    clarity                 NUMERIC(5,2),
    distractor_quality      NUMERIC(5,2),
    single_correctness      NUMERIC(5,2),
    topic_relevance         NUMERIC(5,2),
    difficulty_match        NUMERIC(5,2),
    bloom_match             NUMERIC(5,2),
    option_conciseness      NUMERIC(5,2),
    notation_validity       NUMERIC(5,2),
    duplicate_risk          NUMERIC(5,2),
    composite_score         NUMERIC(5,2) NOT NULL,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE mcq_reviews (
    id                  BIGSERIAL PRIMARY KEY,
    mcq_candidate_id    BIGINT NOT NULL REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    reviewer_id         TEXT NOT NULL,
    action              TEXT NOT NULL CHECK (action IN
                            ('approve','reject','edit','regenerate','flag',
                             'change_difficulty','change_bloom')),
    comment             TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_mcq_reviews_candidate ON mcq_reviews(mcq_candidate_id);

CREATE TABLE mcq_versions (
    id                  BIGSERIAL PRIMARY KEY,
    mcq_candidate_id    BIGINT NOT NULL REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    version_number      INTEGER NOT NULL,
    snapshot            JSONB NOT NULL,
    edited_by           TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (mcq_candidate_id, version_number)
);

CREATE TABLE generation_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    generation_job_id   BIGINT NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    metric_name         TEXT NOT NULL,
    metric_value        NUMERIC,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_generation_metrics_job ON generation_metrics(generation_job_id);

-- ---------------------------------------------------------------------
-- [FUTURE] post-deployment exam analytics — schema reserved now so the
-- live schema doesn't need a breaking migration later.
-- ---------------------------------------------------------------------

CREATE TABLE mcq_item_stats (
    id                          BIGSERIAL PRIMARY KEY,
    mcq_candidate_id            BIGINT NOT NULL UNIQUE REFERENCES mcq_candidates(id) ON DELETE CASCADE,
    attempt_count               INTEGER NOT NULL DEFAULT 0,
    correct_count               INTEGER NOT NULL DEFAULT 0,
    incorrect_count             INTEGER NOT NULL DEFAULT 0,
    difficulty_index            NUMERIC(5,4),
    discrimination_index        NUMERIC(5,4),
    option_selection_distribution JSONB,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);
