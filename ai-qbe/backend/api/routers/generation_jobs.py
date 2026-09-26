"""Generation job lifecycle endpoints.

Phase 2 note on async behavior: job execution is dispatched through
backend/workers/tasks.py, which wraps the same run_generation_job()
executor Celery will call in production. With
AIQBE_CELERY_TASK_ALWAYS_EAGER=true (the dev/test default, see
backend/config.py), `.delay()` runs synchronously in-process and the job
is already COMPLETED by the time `start` returns -- there is no real
running/paused window to observe yet. That window becomes meaningful once
a real broker + worker process is deployed (Phase 5's batching also makes
pause/resume land mid-job rather than only between whole-job runs); the
status-transition rules enforced here already assume that eventual
async reality so this endpoint's behavior does not change later.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload

from backend.api.deps import get_academic_port, get_db
from backend.database.academic_port import AcademicDataPort
from backend.database.models import GenerationJob, GenerationJobTopic, MCQCandidate
from backend.domain.enums import TERMINAL_JOB_STATUSES, GenerationJobStatus, MCQStatus
from backend.domain.schemas import GenerationJobCreate, GenerationJobOut, GenerationJobProgressOut
from backend.security.audit import record_audit_event
from backend.security.rbac import CurrentUser, require_any_authenticated_user, require_job_manager
from backend.workers.tasks import run_generation_job_task

router = APIRouter(prefix="/api/generation-jobs", tags=["generation-jobs"])


@router.post("", response_model=GenerationJobOut, status_code=status.HTTP_201_CREATED)
def create_job(
    payload: GenerationJobCreate,
    db: Session = Depends(get_db),
    port: AcademicDataPort = Depends(get_academic_port),
    user: CurrentUser = Depends(require_job_manager),
):
    subject = port.get_subject(payload.external_subject_id)
    if subject is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown subject: {payload.external_subject_id}")

    job_topics: list[GenerationJobTopic] = []
    for topic_in in payload.topics:
        topic_ref = port.get_topic(payload.external_subject_id, topic_in.external_topic_id)
        if topic_ref is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Unknown topic '{topic_in.external_topic_id}' for subject '{payload.external_subject_id}'",
            )
        job_topics.append(
            GenerationJobTopic(
                external_topic_id=topic_ref.external_id,
                topic_label_snapshot=topic_ref.name,
                manual_context=topic_in.manual_context,
                target_count=0,  # computed by the planner at start time
                weight=topic_in.weight,
            )
        )

    from backend.generation.executor import get_or_create_default_prompt_template, get_or_create_generation_model

    model = get_or_create_generation_model(
        db,
        name=payload.model_name,
        provider_type=payload.model_provider_type,
        version=payload.model_version,
        quantization=payload.model_quantization,
        context_window=payload.model_context_window,
    )
    get_or_create_default_prompt_template(db, model.id)

    job = GenerationJob(
        external_subject_id=subject.external_id,
        subject_label_snapshot=subject.name,
        requested_count=payload.requested_count,
        option_count=payload.option_count,
        difficulty_distribution=payload.difficulty_distribution,
        bloom_distribution=payload.bloom_distribution,
        source_policy={"local_docs": False, "internet": False, "manual_context": True},
        internet_enabled=False,
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by=user.username,
        topics=job_topics,
    )
    db.add(job)
    db.flush()
    record_audit_event(
        db, actor=user.username, action="create_generation_job", resource_type="generation_job", resource_id=str(job.id)
    )
    db.refresh(job)
    return job


@router.get("", response_model=list[GenerationJobOut])
def list_jobs(db: Session = Depends(get_db), _user=Depends(require_any_authenticated_user)):
    return db.query(GenerationJob).order_by(GenerationJob.created_at.desc()).all()


def _get_job_or_404(db: Session, job_id: int) -> GenerationJob:
    job = (
        db.query(GenerationJob)
        .options(selectinload(GenerationJob.topics))
        .filter_by(id=job_id)
        .one_or_none()
    )
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Generation job {job_id} not found")
    return job


@router.get("/{job_id}", response_model=GenerationJobOut)
def get_job(job_id: int, db: Session = Depends(get_db), _user=Depends(require_any_authenticated_user)):
    return _get_job_or_404(db, job_id)


@router.get("/{job_id}/progress", response_model=GenerationJobProgressOut)
def get_job_progress(job_id: int, db: Session = Depends(get_db), _user=Depends(require_any_authenticated_user)):
    job = _get_job_or_404(db, job_id)
    candidates = db.query(MCQCandidate).filter_by(generation_job_id=job.id).all()
    pending_review = sum(1 for c in candidates if c.status == MCQStatus.PENDING_REVIEW)
    approved = sum(1 for c in candidates if c.status == MCQStatus.APPROVED)
    return GenerationJobProgressOut(
        id=job.id,
        status=job.status,
        requested_count=job.requested_count,
        generated_count=job.generated_count,
        pending_review_count=pending_review,
        approved_count=approved,
        rejected_count=job.rejected_count,
        duplicate_count=job.duplicate_count,
    )


@router.post("/{job_id}/start", response_model=GenerationJobOut)
def start_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_job_manager),
):
    job = _get_job_or_404(db, job_id)
    if job.status != GenerationJobStatus.QUEUED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job {job_id} is not QUEUED (status={job.status.value})")
    db.commit()  # ensure the job + topics are visible to the (possibly separate-session) task
    record_audit_event(db, actor=user.username, action="start_generation_job", resource_type="generation_job", resource_id=str(job_id))
    db.commit()
    run_generation_job_task.delay(job_id)
    db.refresh(job)
    return job


@router.post("/{job_id}/pause", response_model=GenerationJobOut)
def pause_job(job_id: int, db: Session = Depends(get_db), user: CurrentUser = Depends(require_job_manager)):
    job = _get_job_or_404(db, job_id)
    if job.status != GenerationJobStatus.RUNNING:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job {job_id} is not RUNNING (status={job.status.value})")
    job.status = GenerationJobStatus.PAUSED
    record_audit_event(db, actor=user.username, action="pause_generation_job", resource_type="generation_job", resource_id=str(job_id))
    db.flush()
    return job


@router.post("/{job_id}/resume", response_model=GenerationJobOut)
def resume_job(job_id: int, db: Session = Depends(get_db), user: CurrentUser = Depends(require_job_manager)):
    job = _get_job_or_404(db, job_id)
    if job.status != GenerationJobStatus.PAUSED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job {job_id} is not PAUSED (status={job.status.value})")
    job.status = GenerationJobStatus.QUEUED
    record_audit_event(db, actor=user.username, action="resume_generation_job", resource_type="generation_job", resource_id=str(job_id))
    db.commit()
    run_generation_job_task.delay(job_id)
    db.refresh(job)
    return job


@router.post("/{job_id}/cancel", response_model=GenerationJobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db), user: CurrentUser = Depends(require_job_manager)):
    job = _get_job_or_404(db, job_id)
    if job.status in TERMINAL_JOB_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job {job_id} is already terminal (status={job.status.value})")
    job.status = GenerationJobStatus.CANCELLED
    record_audit_event(db, actor=user.username, action="cancel_generation_job", resource_type="generation_job", resource_id=str(job_id))
    db.flush()
    return job
