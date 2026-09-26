"""SQLAlchemy models for the AI-QBE owned schema.

Mirrors docs/schema.sql exactly (same tables/columns/constraints) so the
ERD in docs/PHASE0-DESIGN.md, the hand-written DDL, and the ORM models
never drift apart. Academic master data (Program/Semester/Subject/Topic)
is NOT modeled here -- it lives behind backend/database/academic_port.py's
AcademicDataPort interface, referenced from these tables only via
`external_*_id` + a label snapshot (see docstring on GenerationJob).

RAG-related tables (AcademicSource, SourceDocument, DocumentChunk,
MCQSource) are defined now, matching the Phase 0 ERD, but are not
populated by any Phase 2 code path -- Phase 2 generates from
manually-supplied context, not retrieval. They exist so Phase 3 adds
behavior, not schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

from backend.database.base import Base
from backend.domain.enums import BloomLevel, DifficultyLevel, GenerationJobStatus, MCQStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JSONVariant(TypeDecorator):
    """JSONB on Postgres, JSON (TEXT-backed) everywhere else (SQLite)."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class GenerationModel(Base):
    __tablename__ = "generation_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    provider_type: Mapped[str]  # 'ollama' | 'llamacpp' | 'openai_compatible'
    version: Mapped[str]
    quantization: Mapped[str | None]
    context_window: Mapped[int]
    vram_estimate_mb: Mapped[int | None]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (UniqueConstraint("name", "version", "quantization"),)


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_key: Mapped[str]
    version: Mapped[int]
    template_text: Mapped[str]
    model_id: Mapped[int | None] = mapped_column(ForeignKey("generation_models.id"))
    temperature: Mapped[float] = mapped_column(default=0.3)
    top_p: Mapped[float] = mapped_column(default=0.9)
    top_k: Mapped[int | None]
    repeat_penalty: Mapped[float | None]
    max_tokens: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (UniqueConstraint("prompt_key", "version"),)


class ApprovedDomain(Base):
    __tablename__ = "approved_domains"

    id: Mapped[int] = mapped_column(primary_key=True)
    domain: Mapped[str] = mapped_column(unique=True)
    priority: Mapped[int] = mapped_column(default=100)
    min_quality_score: Mapped[float] = mapped_column(default=0)
    notes: Mapped[str | None]


class BlockedDomain(Base):
    __tablename__ = "blocked_domains"

    id: Mapped[int] = mapped_column(primary_key=True)
    domain: Mapped[str] = mapped_column(unique=True)
    reason: Mapped[str | None]


class GenerationJob(Base):
    """A single question-bank generation request.

    `external_subject_id` / `subject_label_snapshot` reference the
    university's own academic database via AcademicDataPort -- AI-QBE never
    copies that table. The snapshot is taken at job-creation time so the
    job's own history stays readable even if the subject is later renamed
    or removed upstream (see docs/PHASE0-DESIGN.md section 4.1).
    """

    __tablename__ = "generation_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_subject_id: Mapped[str]
    subject_label_snapshot: Mapped[str]
    requested_count: Mapped[int]
    option_count: Mapped[int] = mapped_column(default=4)
    difficulty_distribution: Mapped[dict] = mapped_column(JSONVariant)
    bloom_distribution: Mapped[dict] = mapped_column(JSONVariant)
    # Phase 5: question-*format* diversity (docs/PHASE0-DESIGN.md section 11),
    # orthogonal to bloom_distribution. Defaults to 100% single_best_answer,
    # which reproduces pre-Phase-5 behavior exactly for existing jobs.
    question_type_distribution: Mapped[dict] = mapped_column(
        JSONVariant, default=lambda: {"single_best_answer": 1.0}
    )
    source_policy: Mapped[dict] = mapped_column(JSONVariant)
    internet_enabled: Mapped[bool] = mapped_column(default=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("generation_models.id"))
    status: Mapped[GenerationJobStatus] = mapped_column(default=GenerationJobStatus.QUEUED)
    created_by: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]
    over_generation_factor: Mapped[float] = mapped_column(default=1.4)
    max_attempts: Mapped[int] = mapped_column(default=3)
    generated_count: Mapped[int] = mapped_column(default=0)
    approved_count: Mapped[int] = mapped_column(default=0)
    rejected_count: Mapped[int] = mapped_column(default=0)
    duplicate_count: Mapped[int] = mapped_column(default=0)

    topics: Mapped[list["GenerationJobTopic"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    candidates: Mapped[list["MCQCandidate"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["GenerationMetric"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("requested_count > 0", name="ck_requested_count_positive"),
        CheckConstraint("option_count BETWEEN 3 AND 5", name="ck_option_count_range"),
        Index("ix_generation_jobs_status", "status"),
    )


class GenerationJobTopic(Base):
    __tablename__ = "generation_job_topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    generation_job_id: Mapped[int] = mapped_column(
        ForeignKey("generation_jobs.id", ondelete="CASCADE")
    )
    external_topic_id: Mapped[str]
    topic_label_snapshot: Mapped[str]
    # Manually-supplied context for Phase 2 (no RAG yet): the admin
    # provides the source material for this topic directly on the job.
    manual_context: Mapped[str | None]
    target_count: Mapped[int]
    weight: Mapped[float] = mapped_column(default=1.0)

    job: Mapped[GenerationJob] = relationship(back_populates="topics")
    candidates: Mapped[list["MCQCandidate"]] = relationship(back_populates="job_topic")

    __table_args__ = (
        CheckConstraint("target_count >= 0", name="ck_target_count_nonnegative"),
        Index("ix_generation_job_topics_job", "generation_job_id"),
    )


class AcademicSource(Base):
    __tablename__ = "academic_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_type: Mapped[str]  # 'upload' | 'url' | 'instructor_provided'
    url: Mapped[str | None]
    title: Mapped[str | None]
    author: Mapped[str | None]
    publisher: Mapped[str | None]
    license: Mapped[str | None]
    retrieval_date: Mapped[datetime] = mapped_column(default=utcnow)
    approved_domain_id: Mapped[int | None] = mapped_column(ForeignKey("approved_domains.id"))
    document_hash: Mapped[str] = mapped_column(unique=True)
    storage_path: Mapped[str | None]


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    academic_source_id: Mapped[int] = mapped_column(
        ForeignKey("academic_sources.id", ondelete="CASCADE")
    )
    mime_type: Mapped[str]
    page_count: Mapped[int | None]
    extracted_text_hash: Mapped[str | None]
    ingestion_status: Mapped[str] = mapped_column(default="pending")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE")
    )
    external_subject_id: Mapped[str]
    external_topic_id: Mapped[str | None]
    chunk_index: Mapped[int]
    page: Mapped[int | None]
    section: Mapped[str | None]
    text: Mapped[str]
    embedding_vector_id: Mapped[str]  # Qdrant point id; vectors live in Qdrant, not here
    token_count: Mapped[int]

    __table_args__ = (Index("ix_document_chunks_topic", "external_subject_id", "external_topic_id"),)


class MCQCandidate(Base):
    __tablename__ = "mcq_candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    generation_job_id: Mapped[int] = mapped_column(
        ForeignKey("generation_jobs.id", ondelete="CASCADE")
    )
    generation_job_topic_id: Mapped[int] = mapped_column(ForeignKey("generation_job_topics.id"))
    prompt_template_id: Mapped[int] = mapped_column(ForeignKey("prompt_templates.id"))
    generation_model_id: Mapped[int] = mapped_column(ForeignKey("generation_models.id"))
    model_version: Mapped[str]
    batch_id: Mapped[str] = mapped_column(default=lambda: str(uuid.uuid4()))
    question_stem: Mapped[str]
    explanation: Mapped[str | None]
    bloom_level: Mapped[BloomLevel]
    difficulty: Mapped[DifficultyLevel]
    question_type: Mapped[str] = mapped_column(default="single_best_answer")
    confidence: Mapped[float | None]
    status: Mapped[MCQStatus] = mapped_column(default=MCQStatus.GENERATED)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    job: Mapped[GenerationJob] = relationship(back_populates="candidates")
    job_topic: Mapped[GenerationJobTopic] = relationship(back_populates="candidates")
    options: Mapped[list["MCQOption"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", order_by="MCQOption.option_index"
    )
    validation_results: Mapped[list["MCQValidationResult"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    reviews: Mapped[list["MCQReview"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    versions: Mapped[list["MCQVersion"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    quality_score: Mapped["MCQQualityScore | None"] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", uselist=False
    )
    sources: Mapped[list["MCQSource"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )

    @property
    def source_chunk_ids(self) -> list[int]:
        """Document chunk ids this candidate was grounded in (RAG mode
        only; empty for manual-context candidates) -- exposed to the API
        schema for the review UI's evidence panel (Phase 8) and for
        citation auditing.
        """
        return [s.document_chunk_id for s in self.sources]

    __table_args__ = (Index("ix_mcq_candidates_job_status", "generation_job_id", "status"),)


class MCQOption(Base):
    __tablename__ = "mcq_options"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE")
    )
    option_index: Mapped[int]
    text: Mapped[str]
    is_correct: Mapped[bool] = mapped_column(default=False)
    latex_flag: Mapped[bool] = mapped_column(default=False)

    candidate: Mapped[MCQCandidate] = relationship(back_populates="options")

    __table_args__ = (UniqueConstraint("mcq_candidate_id", "option_index"),)


class MCQSource(Base):
    __tablename__ = "mcq_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE")
    )
    document_chunk_id: Mapped[int] = mapped_column(ForeignKey("document_chunks.id"))
    relevance_score: Mapped[float | None]

    candidate: Mapped[MCQCandidate] = relationship(back_populates="sources")


class MCQValidationResult(Base):
    __tablename__ = "mcq_validation_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE")
    )
    validator_name: Mapped[str]
    stage: Mapped[str]
    result: Mapped[str]  # 'pass' | 'fail' | 'uncertain'
    score: Mapped[float | None]
    details: Mapped[dict | None] = mapped_column(JSONVariant)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    candidate: Mapped[MCQCandidate] = relationship(back_populates="validation_results")

    __table_args__ = (
        CheckConstraint("result IN ('pass','fail','uncertain')", name="ck_validation_result_enum"),
    )


class MCQQualityScore(Base):
    __tablename__ = "mcq_quality_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE"), unique=True
    )
    factual_correctness: Mapped[float | None]
    source_grounding: Mapped[float | None]
    clarity: Mapped[float | None]
    distractor_quality: Mapped[float | None]
    single_correctness: Mapped[float | None]
    topic_relevance: Mapped[float | None]
    difficulty_match: Mapped[float | None]
    bloom_match: Mapped[float | None]
    option_conciseness: Mapped[float | None]
    notation_validity: Mapped[float | None]
    duplicate_risk: Mapped[float | None]
    composite_score: Mapped[float]
    computed_at: Mapped[datetime] = mapped_column(default=utcnow)

    candidate: Mapped[MCQCandidate] = relationship(back_populates="quality_score")


class MCQReview(Base):
    __tablename__ = "mcq_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE")
    )
    reviewer_id: Mapped[str]
    action: Mapped[str]
    comment: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    candidate: Mapped[MCQCandidate] = relationship(back_populates="reviews")

    __table_args__ = (
        CheckConstraint(
            "action IN ('approve','reject','edit','regenerate','flag',"
            "'change_difficulty','change_bloom')",
            name="ck_review_action_enum",
        ),
    )


class MCQVersion(Base):
    __tablename__ = "mcq_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE")
    )
    version_number: Mapped[int]
    snapshot: Mapped[dict] = mapped_column(JSONVariant)
    edited_by: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    candidate: Mapped[MCQCandidate] = relationship(back_populates="versions")

    __table_args__ = (UniqueConstraint("mcq_candidate_id", "version_number"),)


class GenerationMetric(Base):
    __tablename__ = "generation_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    generation_job_id: Mapped[int] = mapped_column(
        ForeignKey("generation_jobs.id", ondelete="CASCADE")
    )
    metric_name: Mapped[str]
    metric_value: Mapped[float | None]
    recorded_at: Mapped[datetime] = mapped_column(default=utcnow)

    job: Mapped[GenerationJob] = relationship(back_populates="metrics")


class MCQItemStats(Base):
    """[FUTURE] post-deployment exam analytics -- schema reserved now
    (docs/PHASE0-DESIGN.md section 4.2) so it doesn't require a breaking
    migration later. Not populated by any Phase 2 code path.
    """

    __tablename__ = "mcq_item_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    mcq_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("mcq_candidates.id", ondelete="CASCADE"), unique=True
    )
    attempt_count: Mapped[int] = mapped_column(default=0)
    correct_count: Mapped[int] = mapped_column(default=0)
    incorrect_count: Mapped[int] = mapped_column(default=0)
    difficulty_index: Mapped[float | None]
    discrimination_index: Mapped[float | None]
    option_selection_distribution: Mapped[dict | None] = mapped_column(JSONVariant)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class AuditLogEntry(Base):
    """Security audit trail -- separate from MCQReview (the academic
    record) per docs/PHASE0-DESIGN.md section 14, so the security audit
    survives even if academic review history is pruned or exported out.
    """

    __tablename__ = "audit_log_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str]
    action: Mapped[str]
    resource_type: Mapped[str]
    resource_id: Mapped[str | None]
    details: Mapped[dict | None] = mapped_column(JSONVariant)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class User(Base):
    """Local-auth fallback user store (docs/PHASE0-DESIGN.md section 14:
    "local username/password fallback for MVP"). A production deployment
    with an institutional OIDC/IdP replaces this with token introspection
    against that IdP; the RBAC layer (backend/security/rbac.py) consumes
    a role string either way, so swapping auth backends doesn't touch it.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[str]
    role: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
