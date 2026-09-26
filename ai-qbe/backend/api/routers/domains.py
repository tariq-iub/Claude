"""Approved/blocked domain management (docs/PHASE0-DESIGN.md section 8).

Administrator-only: this list is exactly the boundary between "nothing is
auto-approved" and controlled Internet retrieval, so only Administrator
can extend it (Role matrix, docs/PHASE0-DESIGN.md section 14).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.database.models import ApprovedDomain, BlockedDomain
from backend.domain.enums import Role
from backend.security.audit import record_audit_event
from backend.security.rbac import CurrentUser, require_any_authenticated_user, require_roles

router = APIRouter(prefix="/api/domains", tags=["domains"])
require_admin = require_roles(Role.ADMINISTRATOR)


class ApprovedDomainIn(BaseModel):
    domain: str
    priority: int = 100
    min_quality_score: float = 0.0
    notes: str | None = None


class BlockedDomainIn(BaseModel):
    domain: str
    reason: str | None = None


@router.get("/approved")
def list_approved(db: Session = Depends(get_db), _user: CurrentUser = Depends(require_any_authenticated_user)):
    rows = db.query(ApprovedDomain).order_by(ApprovedDomain.priority.desc()).all()
    return [
        {"id": r.id, "domain": r.domain, "priority": r.priority, "min_quality_score": r.min_quality_score, "notes": r.notes}
        for r in rows
    ]


@router.post("/approved", status_code=status.HTTP_201_CREATED)
def add_approved(
    payload: ApprovedDomainIn, db: Session = Depends(get_db), user: CurrentUser = Depends(require_admin)
):
    domain = payload.domain.lower().strip()
    if db.query(BlockedDomain).filter_by(domain=domain).one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"'{domain}' is on the blocked list; remove it there first")
    existing = db.query(ApprovedDomain).filter_by(domain=domain).one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"'{domain}' is already approved")

    row = ApprovedDomain(domain=domain, priority=payload.priority, min_quality_score=payload.min_quality_score, notes=payload.notes)
    db.add(row)
    db.flush()
    record_audit_event(db, actor=user.username, action="approve_domain", resource_type="approved_domain", resource_id=domain)
    return {"id": row.id, "domain": row.domain}


@router.delete("/approved/{domain}", status_code=status.HTTP_204_NO_CONTENT)
def remove_approved(domain: str, db: Session = Depends(get_db), user: CurrentUser = Depends(require_admin)):
    row = db.query(ApprovedDomain).filter_by(domain=domain.lower()).one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"'{domain}' is not on the approved list")
    db.delete(row)
    record_audit_event(db, actor=user.username, action="unapprove_domain", resource_type="approved_domain", resource_id=domain)


@router.get("/blocked")
def list_blocked(db: Session = Depends(get_db), _user: CurrentUser = Depends(require_any_authenticated_user)):
    rows = db.query(BlockedDomain).all()
    return [{"id": r.id, "domain": r.domain, "reason": r.reason} for r in rows]


@router.post("/blocked", status_code=status.HTTP_201_CREATED)
def add_blocked(payload: BlockedDomainIn, db: Session = Depends(get_db), user: CurrentUser = Depends(require_admin)):
    domain = payload.domain.lower().strip()
    # Blocking always wins: if it was approved, remove that first so the
    # two lists never disagree about the same domain (DomainPolicy checks
    # blocked before approved, but keeping both lists consistent avoids
    # confusing audit trails).
    approved = db.query(ApprovedDomain).filter_by(domain=domain).one_or_none()
    if approved is not None:
        db.delete(approved)

    existing = db.query(BlockedDomain).filter_by(domain=domain).one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"'{domain}' is already blocked")

    row = BlockedDomain(domain=domain, reason=payload.reason)
    db.add(row)
    db.flush()
    record_audit_event(db, actor=user.username, action="block_domain", resource_type="blocked_domain", resource_id=domain)
    return {"id": row.id, "domain": row.domain}


@router.delete("/blocked/{domain}", status_code=status.HTTP_204_NO_CONTENT)
def remove_blocked(domain: str, db: Session = Depends(get_db), user: CurrentUser = Depends(require_admin)):
    row = db.query(BlockedDomain).filter_by(domain=domain.lower()).one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"'{domain}' is not on the blocked list")
    db.delete(row)
    record_audit_event(db, actor=user.username, action="unblock_domain", resource_type="blocked_domain", resource_id=domain)
