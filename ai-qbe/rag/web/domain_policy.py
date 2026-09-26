"""Domain governance: ApprovedDomain / BlockedDomain / SourcePriority /
MinimumSourceQuality, per docs/PHASE0-DESIGN.md section 8 and master
prompt section 6.

Deny-by-default: a domain that is neither explicitly approved nor
explicitly blocked is still rejected. "Nothing is auto-approved" (master
prompt section 36) is enforced here structurally, not by convention --
`is_allowed()` has no code path that returns True without an
`ApprovedDomain` row existing for that host.
"""

from __future__ import annotations

import dataclasses
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from backend.database.models import ApprovedDomain, BlockedDomain


@dataclasses.dataclass
class DomainDecision:
    allowed: bool
    reason: str
    priority: int | None = None
    min_quality_score: float | None = None


def _extract_host(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower()


class DomainPolicy:
    """Reads ApprovedDomain/BlockedDomain from the database. A thin class
    (not a bare function) so it can be swapped for a cached/precompiled
    implementation later without touching call sites -- domain checks run
    on every candidate URL a search adapter returns, so this is a
    plausible future hot path even though a per-call DB query is fine at
    Phase 4's scale.
    """

    def __init__(self, db: Session):
        self._db = db

    def evaluate(self, url: str) -> DomainDecision:
        host = _extract_host(url)
        if not host:
            return DomainDecision(False, "unparseable_url")

        blocked = self._db.query(BlockedDomain).filter_by(domain=host).one_or_none()
        if blocked is not None:
            return DomainDecision(False, f"domain_blocked: {blocked.reason or 'no reason recorded'}")

        approved = self._db.query(ApprovedDomain).filter_by(domain=host).one_or_none()
        if approved is None:
            return DomainDecision(False, "domain_not_in_approved_list")

        return DomainDecision(
            True, "approved", priority=approved.priority, min_quality_score=approved.min_quality_score
        )
