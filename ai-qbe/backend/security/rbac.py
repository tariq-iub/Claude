"""FastAPI dependencies enforcing role-based access control.

Centralizes the access matrix (docs/PHASE0-DESIGN.md section 14) so a
route's required role is declared once, at the route, rather than
re-implemented per-handler.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.domain.enums import ROLES_ALLOWED_TO_MANAGE_JOBS, ROLES_ALLOWED_TO_REVIEW, Role
from backend.security.auth import InvalidTokenError, decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


@dataclass(frozen=True)
class CurrentUser:
    username: str
    role: Role


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    try:
        payload = decode_access_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc
    try:
        role = Role(payload["role"])
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown role in token") from exc
    return CurrentUser(username=payload["sub"], role=role)


def require_roles(*allowed: Role):
    def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{user.role.value}' is not permitted to perform this action.",
            )
        return user

    return dependency


require_job_manager = require_roles(*ROLES_ALLOWED_TO_MANAGE_JOBS)
require_reviewer = require_roles(*ROLES_ALLOWED_TO_REVIEW)
require_any_authenticated_user = get_current_user
