"""Question (MCQ candidate) review endpoints.

This is the Phase 2 slice of what becomes the full Phase 8 human-review
system: single-item approve/reject/edit/regenerate, with every action
versioned (mcq_versions) and audited (mcq_reviews). Bulk review, filters
beyond the basics, and the MathJax-rendering UI itself are Phase 8 scope.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from backend.api.deps import get_db
from backend.database.models import GenerationJob, MCQCandidate, MCQOption, MCQReview, MCQVersion
from backend.domain.enums import MCQStatus, ReviewAction
from backend.domain.schemas import MCQCandidateOut, ReviewActionIn
from backend.security.rbac import CurrentUser, require_any_authenticated_user, require_reviewer

router = APIRouter(prefix="/api/questions", tags=["questions"])


def _get_candidate_or_404(db: Session, question_id: int) -> MCQCandidate:
    candidate = (
        db.query(MCQCandidate)
        .options(selectinload(MCQCandidate.options), selectinload(MCQCandidate.validation_results))
        .filter_by(id=question_id)
        .one_or_none()
    )
    if candidate is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Question {question_id} not found")
    return candidate


def _snapshot(candidate: MCQCandidate) -> dict:
    return {
        "question_stem": candidate.question_stem,
        "explanation": candidate.explanation,
        "difficulty": candidate.difficulty.value,
        "bloom_level": candidate.bloom_level.value,
        "status": candidate.status.value,
        "options": [{"index": o.option_index, "text": o.text, "is_correct": o.is_correct} for o in candidate.options],
    }


def _next_version_number(db: Session, candidate_id: int) -> int:
    last = (
        db.query(MCQVersion)
        .filter_by(mcq_candidate_id=candidate_id)
        .order_by(MCQVersion.version_number.desc())
        .first()
    )
    return (last.version_number + 1) if last else 1


@router.get("", response_model=list[MCQCandidateOut])
def list_questions(
    generation_job_id: int | None = None,
    status_filter: MCQStatus | None = None,
    db: Session = Depends(get_db),
    _user: CurrentUser = Depends(require_any_authenticated_user),
):
    query = db.query(MCQCandidate).options(selectinload(MCQCandidate.options))
    if generation_job_id is not None:
        query = query.filter(MCQCandidate.generation_job_id == generation_job_id)
    if status_filter is not None:
        query = query.filter(MCQCandidate.status == status_filter)
    return query.order_by(MCQCandidate.created_at.desc()).all()


@router.get("/{question_id}", response_model=MCQCandidateOut)
def get_question(question_id: int, db: Session = Depends(get_db), _user: CurrentUser = Depends(require_any_authenticated_user)):
    return _get_candidate_or_404(db, question_id)


@router.post("/{question_id}/approve", response_model=MCQCandidateOut)
def approve_question(question_id: int, db: Session = Depends(get_db), user: CurrentUser = Depends(require_reviewer)):
    candidate = _get_candidate_or_404(db, question_id)
    if candidate.status != MCQStatus.PENDING_REVIEW:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Question {question_id} is not PENDING_REVIEW")
    candidate.status = MCQStatus.APPROVED
    db.add(MCQReview(mcq_candidate_id=candidate.id, reviewer_id=user.username, action=ReviewAction.APPROVE.value))
    job = db.query(GenerationJob).filter_by(id=candidate.generation_job_id).one()
    job.approved_count += 1
    db.flush()
    return candidate


@router.post("/{question_id}/reject", response_model=MCQCandidateOut)
def reject_question(
    question_id: int,
    payload: ReviewActionIn,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
):
    candidate = _get_candidate_or_404(db, question_id)
    if candidate.status not in (MCQStatus.PENDING_REVIEW, MCQStatus.NEEDS_REVISION):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Question {question_id} cannot be rejected from status {candidate.status.value}")
    candidate.status = MCQStatus.REJECTED
    db.add(
        MCQReview(
            mcq_candidate_id=candidate.id,
            reviewer_id=user.username,
            action=ReviewAction.REJECT.value,
            comment=payload.comment,
        )
    )
    job = db.query(GenerationJob).filter_by(id=candidate.generation_job_id).one()
    job.rejected_count += 1
    db.flush()
    return candidate


@router.post("/{question_id}/edit", response_model=MCQCandidateOut)
def edit_question(
    question_id: int,
    payload: ReviewActionIn,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
):
    candidate = _get_candidate_or_404(db, question_id)
    if candidate.status not in (MCQStatus.PENDING_REVIEW, MCQStatus.NEEDS_REVISION):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Question {question_id} cannot be edited from status {candidate.status.value}")

    db.add(
        MCQVersion(
            mcq_candidate_id=candidate.id,
            version_number=_next_version_number(db, candidate.id),
            snapshot=_snapshot(candidate),
            edited_by=user.username,
        )
    )

    if payload.edited_question_stem is not None:
        candidate.question_stem = payload.edited_question_stem
    if payload.edited_options is not None:
        if payload.edited_correct_option is None or not (0 <= payload.edited_correct_option < len(payload.edited_options)):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "edited_correct_option must index into edited_options")
        for opt in list(candidate.options):
            db.delete(opt)
        db.flush()
        for idx, text in enumerate(payload.edited_options):
            db.add(
                MCQOption(
                    mcq_candidate_id=candidate.id,
                    option_index=idx,
                    text=text,
                    is_correct=(idx == payload.edited_correct_option),
                )
            )
    if payload.new_difficulty is not None:
        candidate.difficulty = payload.new_difficulty
    if payload.new_bloom_level is not None:
        candidate.bloom_level = payload.new_bloom_level

    db.add(
        MCQReview(
            mcq_candidate_id=candidate.id,
            reviewer_id=user.username,
            action=ReviewAction.EDIT.value,
            comment=payload.comment,
        )
    )
    db.flush()
    db.refresh(candidate)
    return candidate


@router.post("/{question_id}/flag", response_model=MCQCandidateOut)
def flag_question(
    question_id: int,
    payload: ReviewActionIn,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
):
    candidate = _get_candidate_or_404(db, question_id)
    db.add(
        MCQReview(
            mcq_candidate_id=candidate.id,
            reviewer_id=user.username,
            action=ReviewAction.FLAG.value,
            comment=payload.comment,
        )
    )
    db.flush()
    return candidate


@router.post("/{question_id}/regenerate", response_model=MCQCandidateOut)
def regenerate_question(
    question_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_reviewer),
):
    """Marks the current candidate as superseded and generates a fresh one
    for the same job topic / Bloom level / difficulty / context.
    """
    from backend.database.models import GenerationModel, PromptTemplate
    from backend.generation.executor import _generate_one  # local import avoids a circular import at module load
    from backend.generation.planner import PlanCell
    from llm.providers.factory import build_provider

    candidate = _get_candidate_or_404(db, question_id)
    job = db.query(GenerationJob).filter_by(id=candidate.generation_job_id).one()
    topic = candidate.job_topic
    prompt_template = db.query(PromptTemplate).filter_by(id=candidate.prompt_template_id).one()

    candidate.status = MCQStatus.REJECTED
    db.add(
        MCQReview(
            mcq_candidate_id=candidate.id,
            reviewer_id=user.username,
            action=ReviewAction.REGENERATE.value,
            comment="Superseded by regeneration",
        )
    )
    db.flush()

    generation_model = db.query(GenerationModel).filter_by(id=job.model_id).one()
    provider = build_provider(
        generation_model.provider_type,
        generation_model.name,
        model_version=generation_model.version,
        quantization=generation_model.quantization,
        context_window=generation_model.context_window,
    )

    cell = PlanCell(topic.id, candidate.bloom_level.value, candidate.difficulty.value, 1)
    new_candidate, _ = _generate_one(db, job, topic, provider, prompt_template, cell)
    db.flush()
    db.refresh(new_candidate)
    return new_candidate
