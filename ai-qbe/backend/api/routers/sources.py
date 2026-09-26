"""Document ingestion endpoints: instructor-provided uploads (Phase 3) and
approved-domain web ingestion (Phase 4). Both land in the same
`academic_sources` / `document_chunks` tables -- only how the bytes
arrive differs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import (
    get_academic_port,
    get_db,
    get_domain_policy,
    get_embedding_provider,
    get_search_adapter,
    get_vector_store,
    get_web_fetcher,
)
from backend.database.academic_port import AcademicDataPort
from backend.database.models import AcademicSource, DocumentChunk, SourceDocument
from backend.embeddings.base import IEmbeddingProvider
from backend.security.audit import record_audit_event
from backend.security.rbac import CurrentUser, require_any_authenticated_user, require_job_manager
from rag.ingestion import ingest_document
from rag.vectorstore import IVectorStore
from rag.web.domain_policy import DomainPolicy
from rag.web.fetch import IWebFetcher
from rag.web.search import ISearchAdapter
from rag.web.ingestion import ingest_from_url

router = APIRouter(prefix="/api/sources", tags=["sources"])

_MIME_BY_EXTENSION = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


@router.post("", status_code=status.HTTP_201_CREATED)
def upload_source(
    external_subject_id: str = Form(...),
    external_topic_id: str | None = Form(None),
    title: str = Form(...),
    author: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    port: AcademicDataPort = Depends(get_academic_port),
    vector_store: IVectorStore = Depends(get_vector_store),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    user: CurrentUser = Depends(require_job_manager),
):
    if port.get_subject(external_subject_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown subject: {external_subject_id}")
    if external_topic_id is not None and port.get_topic(external_subject_id, external_topic_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown topic: {external_topic_id}")

    extension = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    mime_type = _MIME_BY_EXTENSION.get(extension)
    if mime_type is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{extension}'. Supported: {sorted(_MIME_BY_EXTENSION)}",
        )

    file_bytes = file.file.read()
    # Phase 0 section 34 size cap: reject anything absurd before it reaches
    # extraction/chunking. 50MB is generous for course PDFs/notes.
    max_bytes = 50 * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_bytes} byte limit")

    source = ingest_document(
        db,
        file_bytes=file_bytes,
        mime_type=mime_type,
        title=title,
        external_subject_id=external_subject_id,
        external_topic_id=external_topic_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        author=author,
    )
    record_audit_event(
        db, actor=user.username, action="upload_source", resource_type="academic_source", resource_id=str(source.id)
    )
    db.flush()

    chunk_count = db.query(DocumentChunk).join(SourceDocument).filter(
        SourceDocument.academic_source_id == source.id
    ).count()

    return {
        "id": source.id,
        "title": source.title,
        "document_hash": source.document_hash,
        "chunk_count": chunk_count,
    }


class FromUrlIn(BaseModel):
    url: str
    external_subject_id: str
    external_topic_id: str | None = None
    title: str | None = None


@router.post("/from-url", status_code=status.HTTP_201_CREATED)
def ingest_url_source(
    payload: FromUrlIn,
    db: Session = Depends(get_db),
    port: AcademicDataPort = Depends(get_academic_port),
    domain_policy: DomainPolicy = Depends(get_domain_policy),
    fetcher: IWebFetcher = Depends(get_web_fetcher),
    vector_store: IVectorStore = Depends(get_vector_store),
    embedding_provider: IEmbeddingProvider = Depends(get_embedding_provider),
    user: CurrentUser = Depends(require_job_manager),
):
    """Fetches and ingests one approved-domain URL (docs/PHASE0-DESIGN.md
    section 8). Rejects with 403 if the URL's domain isn't on the
    approved list -- there is no path in this endpoint that ingests an
    unapproved domain, matching "nothing is auto-approved".
    """
    if port.get_subject(payload.external_subject_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown subject: {payload.external_subject_id}")
    if payload.external_topic_id is not None and port.get_topic(payload.external_subject_id, payload.external_topic_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown topic: {payload.external_topic_id}")

    result = ingest_from_url(
        db,
        payload.url,
        external_subject_id=payload.external_subject_id,
        external_topic_id=payload.external_topic_id,
        domain_policy=domain_policy,
        fetcher=fetcher,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        title=payload.title,
    )
    if not result.accepted:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"URL rejected: {result.reason}")

    record_audit_event(
        db,
        actor=user.username,
        action="ingest_url_source",
        resource_type="academic_source",
        resource_id=str(result.source.id) if result.source else None,
        details={"url": payload.url, "reason": result.reason, "injection_findings_count": len(result.injection_findings)},
    )
    db.flush()

    return {
        "id": result.source.id,
        "title": result.source.title,
        "url": result.source.url,
        "document_hash": result.source.document_hash,
        "chunk_count": result.chunk_count,
        "reason": result.reason,
        "injection_findings_count": len(result.injection_findings),
    }


class SearchQueryIn(BaseModel):
    query: str
    max_results: int = 10


@router.post("/search-preview")
def search_preview(
    payload: SearchQueryIn,
    db: Session = Depends(get_db),
    domain_policy: DomainPolicy = Depends(get_domain_policy),
    search_adapter: ISearchAdapter = Depends(get_search_adapter),
    _user: CurrentUser = Depends(require_job_manager),
):
    """Runs the configured search adapter and annotates each result with
    whether its domain is currently approved -- a preview/discovery step
    an administrator uses before deciding what to approve and ingest.
    This endpoint never ingests anything by itself.
    """
    results = search_adapter.search(payload.query, max_results=payload.max_results)
    return [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "domain_decision": domain_policy.evaluate(r.url).reason,
        }
        for r in results
    ]


@router.get("/{source_id}")
def get_source(
    source_id: int, db: Session = Depends(get_db), _user: CurrentUser = Depends(require_any_authenticated_user)
):
    source = db.query(AcademicSource).filter_by(id=source_id).one_or_none()
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Source {source_id} not found")
    documents = db.query(SourceDocument).filter_by(academic_source_id=source.id).all()
    chunk_count = sum(
        db.query(DocumentChunk).filter_by(source_document_id=d.id).count() for d in documents
    )
    return {
        "id": source.id,
        "title": source.title,
        "author": source.author,
        "source_type": source.source_type,
        "document_hash": source.document_hash,
        "retrieval_date": source.retrieval_date,
        "chunk_count": chunk_count,
        "ingestion_status": documents[0].ingestion_status if documents else "unknown",
    }
