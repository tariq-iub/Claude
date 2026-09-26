"""Domain enums shared by the database models, API schemas, and business
logic. Kept in one module so a status name is spelled the same way in the
DB, the API contract, and the generation/validation code -- a drifted
string literal here is exactly how a lifecycle bug hides for months.

Values mirror docs/schema.sql and docs/PHASE0-DESIGN.md sections 11/26.
"""

from __future__ import annotations

import enum


class GenerationJobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PREPARING_KNOWLEDGE = "PREPARING_KNOWLEDGE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Terminal states a job cannot be started/paused/resumed/cancelled out of.
TERMINAL_JOB_STATUSES = {
    GenerationJobStatus.COMPLETED,
    GenerationJobStatus.FAILED,
    GenerationJobStatus.CANCELLED,
}


class MCQStatus(str, enum.Enum):
    GENERATED = "GENERATED"
    STRUCTURE_VALIDATED = "STRUCTURE_VALIDATED"
    FACT_VALIDATED = "FACT_VALIDATED"
    DEDUPLICATED = "DEDUPLICATED"
    QUALITY_CHECKED = "QUALITY_CHECKED"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVISION = "NEEDS_REVISION"
    DUPLICATE = "DUPLICATE"
    INVALID = "INVALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


# Statuses that mean "still alive in the pipeline" vs. terminal-non-approved.
TERMINAL_REJECTED_STATUSES = {
    MCQStatus.REJECTED,
    MCQStatus.DUPLICATE,
    MCQStatus.INVALID,
}


class BloomLevel(str, enum.Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"


class DifficultyLevel(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewAction(str, enum.Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    REGENERATE = "regenerate"
    FLAG = "flag"
    CHANGE_DIFFICULTY = "change_difficulty"
    CHANGE_BLOOM = "change_bloom"


class Role(str, enum.Enum):
    """RBAC roles, per docs/PHASE0-DESIGN.md section 14 / master prompt section 34."""

    ADMINISTRATOR = "administrator"
    QUESTION_GENERATOR = "question_generator"
    ACADEMIC_REVIEWER = "academic_reviewer"
    SUBJECT_EXPERT = "subject_expert"
    EXAM_CONTROLLER = "exam_controller"
    READ_ONLY = "read_only"


# Which roles may call which class of mutating endpoint. Kept centralized
# rather than scattered per-route so the whole access matrix is auditable
# in one place.
ROLES_ALLOWED_TO_MANAGE_JOBS = {Role.ADMINISTRATOR, Role.QUESTION_GENERATOR}
ROLES_ALLOWED_TO_REVIEW = {Role.ADMINISTRATOR, Role.ACADEMIC_REVIEWER, Role.SUBJECT_EXPERT}
ROLES_ALLOWED_READ_ONLY = set(Role)  # every role can read
