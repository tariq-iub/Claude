"""Local username/password auth + JWT issuance (MVP fallback -- see
docs/PHASE0-DESIGN.md section 14: production deployments with an
institutional IdP swap this for OIDC token verification without changing
backend/security/rbac.py, which only ever consumes a role string).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from backend.config import settings

# Using the `bcrypt` library directly rather than passlib's CryptContext:
# passlib 1.7.x's bcrypt backend self-test is incompatible with bcrypt>=4.1
# (raises on its own 72-byte test string), a known upstream issue with no
# passlib release fix at time of writing. bcrypt's own API is stable and
# sufficient for the hash/verify operations this module needs.
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    truncated = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(truncated, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    truncated = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.checkpw(truncated, hashed.encode("utf-8"))


def create_access_token(*, subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


class InvalidTokenError(Exception):
    pass


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
