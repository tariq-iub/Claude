"""FastAPI application factory.

`create_app()` wires the DB session factory into api.deps, creates tables
(dev convenience -- production uses the Alembic migration in
backend/database/migrations, see docs/PHASE0-DESIGN.md section 33), seeds
a default admin user if none exists, and mounts the routers.
"""

from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker

from backend.api import deps
from backend.api.routers import auth, generation_jobs, questions, sources, subjects
from backend.config import settings
from backend.database.academic_port import AcademicDataPort, InMemoryAcademicDataPort
from backend.database.base import Base, make_engine, make_session_factory
from backend.database.models import User
from backend.embeddings.base import IEmbeddingProvider
from backend.embeddings.factory import build_embedding_provider
from backend.security.auth import hash_password
from rag.vectorstore import IVectorStore, QdrantVectorStore


def create_app(
    database_url: str | None = None,
    academic_port: AcademicDataPort | None = None,
    vector_store: IVectorStore | None = None,
    embedding_provider: IEmbeddingProvider | None = None,
    create_tables: bool = True,
    seed_admin: bool = True,
) -> FastAPI:
    engine = make_engine(database_url or settings.database_url)
    session_factory: sessionmaker = make_session_factory(engine)

    if create_tables:
        Base.metadata.create_all(engine)

    deps.configure(
        session_factory,
        academic_port or InMemoryAcademicDataPort(),
        vector_store or QdrantVectorStore(settings.vector_store_location),
        embedding_provider or build_embedding_provider(settings.embedding_provider_type),
    )

    if seed_admin:
        _seed_default_admin(session_factory)

    app = FastAPI(
        title="AI Academic Question Bank Engine (AI-QBE)",
        description="Phase 3: document RAG (ingestion, chunking, embedding, retrieval) plus core API.",
        version="0.3.0",
    )
    app.include_router(auth.router)
    app.include_router(subjects.router)
    app.include_router(sources.router)
    app.include_router(generation_jobs.router)
    app.include_router(questions.router)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    return app


def _seed_default_admin(session_factory: sessionmaker) -> None:
    """Dev/first-run convenience only: creates `admin` / `changeme123` with
    the Administrator role if no users exist yet. Production deployments
    must rotate this immediately or disable local auth entirely in favor
    of the institution's OIDC/IdP (docs/PHASE0-DESIGN.md section 14).
    """
    db = session_factory()
    try:
        if db.query(User).count() == 0:
            db.add(User(username="admin", hashed_password=hash_password("changeme123"), role="administrator"))
            db.commit()
    finally:
        db.close()


app = create_app()
