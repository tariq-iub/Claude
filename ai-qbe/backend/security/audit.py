"""Immutable audit trail for state-changing actions (docs/PHASE0-DESIGN.md
section 14). Deliberately a separate table/function from MCQReview (the
academic record) -- security audit must survive independent of academic
data retention decisions.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.database.models import AuditLogEntry


def record_audit_event(
    db: Session, *, actor: str, action: str, resource_type: str, resource_id: str | None, details: dict | None = None
) -> None:
    db.add(
        AuditLogEntry(
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
        )
    )
