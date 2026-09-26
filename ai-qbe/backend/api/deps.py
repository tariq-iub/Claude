"""Shared FastAPI dependencies: DB session and the AcademicDataPort
implementation currently configured for this deployment.
"""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from backend.database.academic_port import AcademicDataPort, InMemoryAcademicDataPort
from backend.embeddings.base import IEmbeddingProvider
from rag.vectorstore import IVectorStore
from rag.web.domain_policy import DomainPolicy
from rag.web.fetch import IWebFetcher, RequestsFetcher
from rag.web.search import ISearchAdapter, NullSearchAdapter

# Overridden at app startup (see api/main.py) once the engine/session
# factory for the configured database_url exists; kept as module-level
# mutables so tests can swap in a SQLite-backed factory / in-memory
# vector store / fake web fetcher without touching route code.
_session_factory: sessionmaker | None = None
_academic_port: AcademicDataPort = InMemoryAcademicDataPort()
_vector_store: IVectorStore | None = None
_embedding_provider: IEmbeddingProvider | None = None
_web_fetcher: IWebFetcher = RequestsFetcher()
_search_adapter: ISearchAdapter = NullSearchAdapter()


def configure(
    session_factory: sessionmaker,
    academic_port: AcademicDataPort | None = None,
    vector_store: IVectorStore | None = None,
    embedding_provider: IEmbeddingProvider | None = None,
    web_fetcher: IWebFetcher | None = None,
    search_adapter: ISearchAdapter | None = None,
) -> None:
    global _session_factory, _academic_port, _vector_store, _embedding_provider, _web_fetcher, _search_adapter
    _session_factory = session_factory
    if academic_port is not None:
        _academic_port = academic_port
    if vector_store is not None:
        _vector_store = vector_store
    if embedding_provider is not None:
        _embedding_provider = embedding_provider
    if web_fetcher is not None:
        _web_fetcher = web_fetcher
    if search_adapter is not None:
        _search_adapter = search_adapter


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


def get_vector_store() -> IVectorStore:
    if _vector_store is None:
        raise RuntimeError("Vector store not configured; call api.deps.configure() first.")
    return _vector_store


def get_embedding_provider() -> IEmbeddingProvider:
    if _embedding_provider is None:
        raise RuntimeError("Embedding provider not configured; call api.deps.configure() first.")
    return _embedding_provider


def get_web_fetcher() -> IWebFetcher:
    return _web_fetcher


def get_search_adapter() -> ISearchAdapter:
    return _search_adapter


def get_domain_policy(db: Session = Depends(get_db)) -> DomainPolicy:
    return DomainPolicy(db)
