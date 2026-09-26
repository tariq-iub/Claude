"""Shared FastAPI dependencies: DB session and the AcademicDataPort
implementation currently configured for this deployment.
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session, sessionmaker

from backend.database.academic_port import AcademicDataPort, InMemoryAcademicDataPort

# Overridden at app startup (see api/main.py) once the engine/session
# factory for the configured database_url exists; kept as a module-level
# mutable so tests can swap in a SQLite-backed factory without touching
# route code.
_session_factory: sessionmaker | None = None
_academic_port: AcademicDataPort = InMemoryAcademicDataPort()


def configure(session_factory: sessionmaker, academic_port: AcademicDataPort | None = None) -> None:
    global _session_factory, _academic_port
    _session_factory = session_factory
    if academic_port is not None:
        _academic_port = academic_port


def get_db() -> Iterator[Session]:
    if _session_factory is None:
        raise RuntimeError("Database session factory not configured; call api.deps.configure() first.")
    db = _session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_academic_port() -> AcademicDataPort:
    return _academic_port
