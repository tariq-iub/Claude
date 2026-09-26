"""Read-only endpoints over the AcademicDataPort -- never a direct query
against a university table (see backend/database/academic_port.py).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.deps import get_academic_port
from backend.database.academic_port import AcademicDataPort
from backend.domain.schemas import SubjectOut, TopicOut
from backend.security.rbac import require_any_authenticated_user

router = APIRouter(prefix="/api/subjects", tags=["subjects"], dependencies=[Depends(require_any_authenticated_user)])


@router.get("", response_model=list[SubjectOut])
def list_subjects(port: AcademicDataPort = Depends(get_academic_port)):
    return [SubjectOut(**s.__dict__) for s in port.list_subjects()]


@router.get("/{subject_id}/topics", response_model=list[TopicOut])
def list_topics(subject_id: str, port: AcademicDataPort = Depends(get_academic_port)):
    if port.get_subject(subject_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown subject: {subject_id}")
    topics = port.list_topics(subject_id)
    return [
        TopicOut(
            external_id=t.external_id,
            subject_external_id=t.subject_external_id,
            name=t.name,
            subtopics=list(t.subtopics),
        )
        for t in topics
    ]
