"""Celery tasks. Each task opens its own DB session (workers are separate
processes from the API in production; sharing a session across process
boundaries isn't possible) and delegates all real logic to
backend/generation/executor.py -- tasks themselves stay thin so the same
executor logic is exercised identically whether called via Celery
(production) or directly in a test.
"""

from __future__ import annotations

import logging

from backend.database.models import GenerationJob, GenerationModel, PromptTemplate
from backend.generation.executor import run_generation_job
from backend.workers.celery_app import celery_app
from llm.providers.factory import build_provider

logger = logging.getLogger(__name__)


@celery_app.task(name="aiqbe.run_generation_job")
def run_generation_job_task(job_id: int) -> None:
    from backend.api import deps  # re-import to read the current module-level factory

    if deps._session_factory is None:
        raise RuntimeError("Session factory not configured; call backend.api.deps.configure() first.")

    db = deps._session_factory()
    try:
        job = db.get(GenerationJob, job_id)
        if job is None:
            logger.error("run_generation_job_task: job %s not found", job_id)
            return

        generation_model = db.get(GenerationModel, job.model_id)
        provider = build_provider(
            generation_model.provider_type,
            generation_model.name,
            model_version=generation_model.version,
            quantization=generation_model.quantization,
            context_window=generation_model.context_window,
        )
        prompt_template = (
            db.query(PromptTemplate).filter_by(prompt_key="mcq_generation", version=1).one()
        )

        run_generation_job(db, job, provider, prompt_template)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
