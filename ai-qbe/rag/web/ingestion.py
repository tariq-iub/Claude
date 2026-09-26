"""Approved-domain web ingestion: search/URL -> domain policy -> fetch ->
sanitize -> injection-scrub -> the same chunk/embed/store pipeline
`rag.ingestion` uses for uploads.

This is where Phase 4's controlled Internet research actually lands
evidence in the system: once ingested, a web-sourced chunk is
indistinguishable in *retrieval* from an uploaded one (same
`document_chunks` table, same vector-store payload shape) -- Phase 3's
Topic Knowledge Pack retrieval and the generation executor's RAG mode
need no changes to pick up web-sourced evidence. What Phase 4 adds is
entirely on the ingestion side: nothing gets into `document_chunks`
without first clearing `DomainPolicy`, and nothing that clears it is
trusted as instructions rather than data (rag/web/injection_defense.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from backend.database.models import AcademicSource, ApprovedDomain, SourceDocument
from backend.embeddings.base import IEmbeddingProvider
from rag.extraction import ExtractedPage, extract_pdf
from rag.ingestion import ingest_pages
from rag.vectorstore import IVectorStore
from rag.web.domain_policy import DomainPolicy
from rag.web.fetch import IWebFetcher
from rag.web.injection_defense import scrub_untrusted_text
from rag.web.sanitize import sanitize_html


@dataclasses.dataclass
class WebIngestionResult:
    accepted: bool
    reason: str
    source: AcademicSource | None = None
    chunk_count: int = 0
    injection_findings: list[str] = dataclasses.field(default_factory=list)


def ingest_from_url(
    db: Session,
    url: str,
    *,
    external_subject_id: str,
    external_topic_id: str | None,
    domain_policy: DomainPolicy,
    fetcher: IWebFetcher,
    embedding_provider: IEmbeddingProvider,
    vector_store: IVectorStore,
    title: str | None = None,
) -> WebIngestionResult:
    decision = domain_policy.evaluate(url)
    if not decision.allowed:
        return WebIngestionResult(accepted=False, reason=decision.reason)

    fetch_result = fetcher.fetch(url)
    if not fetch_result.ok:
        return WebIngestionResult(accepted=False, reason=fetch_result.rejected_reason or "fetch_failed")

    document_hash = hashlib.sha256(fetch_result.body).hexdigest()
    existing = db.query(AcademicSource).filter_by(document_hash=document_hash).one_or_none()
    if existing is not None:
        return WebIngestionResult(accepted=True, reason="already_ingested", source=existing)

    injection_findings: list[str] = []
    if fetch_result.content_type == "application/pdf":
        pages = extract_pdf(fetch_result.body)
        # PDFs get the same injection scrub as HTML text -- a malicious
        # actor could embed instruction-like text in a PDF just as easily.
        scrubbed_pages = []
        for page in pages:
            scrub = scrub_untrusted_text(page.text)
            injection_findings.extend(scrub.flagged_lines)
            scrubbed_pages.append(ExtractedPage(page=page.page, text=scrub.cleaned_text))
        pages = scrubbed_pages
        page_title = title
    else:  # text/html, enforced by DomainPolicy + fetcher's content-type allowlist
        sanitized = sanitize_html(fetch_result.body.decode("utf-8", errors="replace"))
        scrub = scrub_untrusted_text(sanitized.text)
        injection_findings.extend(scrub.flagged_lines)
        pages = [ExtractedPage(page=None, text=scrub.cleaned_text)]
        page_title = title or sanitized.title or url

    approved_domain = (
        db.query(ApprovedDomain).filter_by(domain=(urlparse(url).hostname or "").lower()).one_or_none()
    )

    source = AcademicSource(
        source_type="url",
        url=url,
        title=page_title or url,
        publisher=urlparse(url).hostname,
        retrieval_date=datetime.now(timezone.utc),
        approved_domain_id=approved_domain.id if approved_domain else None,
        document_hash=document_hash,
    )
    db.add(source)
    db.flush()

    source_document = SourceDocument(
        academic_source_id=source.id,
        mime_type=fetch_result.content_type or "text/html",
        ingestion_status="processing",
    )
    db.add(source_document)
    db.flush()

    chunk_count = ingest_pages(
        db,
        source_document=source_document,
        pages=pages,
        external_subject_id=external_subject_id,
        external_topic_id=external_topic_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return WebIngestionResult(
        accepted=True,
        reason="ingested",
        source=source,
        chunk_count=chunk_count,
        injection_findings=injection_findings,
    )
