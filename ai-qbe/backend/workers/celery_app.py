"""Celery application. Task-level code lives in tasks.py; this module
only wires up the broker/backend and eager-mode setting from config, so
`celery -A backend.workers.celery_app worker` and the API's in-process
`.delay()` calls (when eager) both go through the exact same setup.
"""

from __future__ import annotations

from celery import Celery

from backend.config import settings

celery_app = Celery(
    "aiqbe",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_always_eager = settings.celery_task_always_eager
celery_app.conf.task_eager_propagates = True
