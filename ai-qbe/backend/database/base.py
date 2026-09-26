"""Declarative base + session/engine factory.

Dialect-agnostic on purpose: Postgres in production (per
docs/PHASE0-DESIGN.md section 1), SQLite for tests and for running this
repo's own test suite without standing up a Postgres server. SQLAlchemy's
`Enum` type degrades to VARCHAR+CHECK on SQLite and a native ENUM on
Postgres automatically, so the same model definitions work on both.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, future=True)


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
