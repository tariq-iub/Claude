"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.domain.enums import (
    BloomLevel,
    DifficultyLevel,
    GenerationJobStatus,
    MCQStatus,
    QuestionType,
    ReviewAction,
)

# --- Auth -------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


# --- Academic data (read-only, via AcademicDataPort) -------------------


class SubjectOut(BaseModel):
    external_id: str
    name: str
    program: str
    semester: str


class TopicOut(BaseModel):
    external_id: str
    subject_external_id: str
    name: str
    subtopics: list[str] = []


# --- Generation jobs -----------------------------------------------------


class GenerationJobTopicIn(BaseModel):
    external_topic_id: str
    manual_context: str | None = Field(
        None,
        description=(
            "Manually-supplied source context (Phase 2 mode). Required if the "
            "job's use_rag is false; optional (and merged in as extra context) "
            "if use_rag is true."
        ),
    )
    weight: float = 1.0


class GenerationJobCreate(BaseModel):
    external_subject_id: str
    requested_count: int = Field(..., gt=0)
    option_count: int = Field(4, ge=3, le=5)
    difficulty_distribution: dict[str, float] = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    bloom_distribution: dict[str, float] = {
        "remember": 0.25,
        "understand": 0.3,
        "apply": 0.3,
        "analyze": 0.15,
    }
    question_type_distribution: dict[str, float] = {
        "single_best_answer": 0.6,
        "scenario_based": 0.25,
        "negative": 0.10,
        "definition_recall": 0.05,
    }
    topics: list[GenerationJobTopicIn]
    use_rag: bool = Field(
        False,
        description=(
            "If true, context is retrieved from ingested documents (Phase 3 "
            "RAG) via the Topic Knowledge Pack, falling back to a topic's "
            "manual_context only if retrieval finds no evidence. If false "
            "(default), every topic must supply manual_context (Phase 2 mode)."
        ),
    )
    model_provider_type: str = "mock"
    model_name: str = "mock-model"
    model_version: str = "unknown"
    model_quantization: str | None = None
    model_context_window: int = 4096

    @model_validator(mode="after")
    def _manual_context_required_unless_rag(self) -> "GenerationJobCreate":
        if not self.use_rag:
            missing = [t.external_topic_id for t in self.topics if not t.manual_context]
            if missing:
                raise ValueError(
                    f"manual_context is required for topics {missing} when use_rag is false"
                )
        return self

    @field_validator("difficulty_distribution", "bloom_distribution", "question_type_distribution")
    @classmethod
    def _fractions_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        total = sum(value.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"distribution must sum to ~1.0, got {total}")
        return value

    @field_validator("bloom_distribution")
    @classmethod
    def _bloom_keys_valid(cls, value: dict[str, float]) -> dict[str, float]:
        valid = {b.value for b in BloomLevel}
        if not set(value.keys()).issubset(valid):
            raise ValueError(f"bloom_distribution keys must be a subset of {valid}")
        return value

    @field_validator("difficulty_distribution")
    @classmethod
    def _difficulty_keys_valid(cls, value: dict[str, float]) -> dict[str, float]:
        valid = {d.value for d in DifficultyLevel}
        if not set(value.keys()).issubset(valid):
            raise ValueError(f"difficulty_distribution keys must be a subset of {valid}")
        return value

    @field_validator("question_type_distribution")
    @classmethod
    def _question_type_keys_valid(cls, value: dict[str, float]) -> dict[str, float]:
        valid = {q.value for q in QuestionType}
        if not set(value.keys()).issubset(valid):
            raise ValueError(f"question_type_distribution keys must be a subset of {valid}")
        return value


class GenerationJobOut(BaseModel):
    id: int
    external_subject_id: str
    subject_label_snapshot: str
    requested_count: int
    option_count: int
    status: GenerationJobStatus
    created_by: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    generated_count: int
    approved_count: int
    rejected_count: int
    duplicate_count: int

    model_config = {"from_attributes": True}


class GenerationJobProgressOut(BaseModel):
    id: int
    status: GenerationJobStatus
    requested_count: int
    generated_count: int
    pending_review_count: int
    approved_count: int
    rejected_count: int
    duplicate_count: int


# --- Questions (MCQ candidates) -----------------------------------------


class MCQOptionOut(BaseModel):
    option_index: int
    text: str
    is_correct: bool

    model_config = {"from_attributes": True}


class MCQValidationResultOut(BaseModel):
    validator_name: str
    stage: str
    result: str
    details: dict | None

    model_config = {"from_attributes": True}


class MCQCandidateOut(BaseModel):
    id: int
    generation_job_id: int
    question_stem: str
    explanation: str | None
    bloom_level: BloomLevel
    difficulty: DifficultyLevel
    question_type: str
    status: MCQStatus
    confidence: float | None
    options: list[MCQOptionOut]
    validation_results: list[MCQValidationResultOut] = []
    source_chunk_ids: list[int] = []

    model_config = {"from_attributes": True}


class ReviewActionIn(BaseModel):
    action: ReviewAction
    comment: str | None = None
    # Only used for action=edit
    edited_question_stem: str | None = None
    edited_options: list[str] | None = None
    edited_correct_option: int | None = None
    # Only used for change_difficulty / change_bloom
    new_difficulty: DifficultyLevel | None = None
    new_bloom_level: BloomLevel | None = None
