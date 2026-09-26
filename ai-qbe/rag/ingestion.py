"""Document ingestion pipeline: upload -> extract -> clean/chunk -> embed
-> upsert into the vector store -> persist DocumentChunk rows with full
metadata. Ties together rag/extraction.py, rag/chunking.py,
backend/embeddings/, and rag/vectorstore.py.

Source documents are hash-deduplicated (docs/schema.sql:
academic_sources.document_hash UNIQUE) so re-uploading the same file for
the same or a different job is a no-op rather than a duplicate ingestion.
"""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from backend.database.models import AcademicSource, DocumentChunk, SourceDocument
from backend.embeddings.base import IEmbeddingProvider
from rag.chunking import semantic_chunk
from rag.extraction import extract_text
from rag.vectorstore import ChunkPoint, IVectorStore


def ingest_document(
    db: Session,
    *,
    file_bytes: bytes,
    mime_type: str,
    title: str,
    external_subject_id: str,
    external_topic_id: str | None,
    embedding_provider: IEmbeddingProvider,
    vector_store: IVectorStore,
    source_type: str = "instructor_provided",
    author: str | None = None,
    publisher: str | None = None,
    license: str | None = None,
) -> AcademicSource:
    document_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = db.query(AcademicSource).filter_by(document_hash=document_hash).one_or_none()
    if existing is not None:
        return existing

    source = AcademicSource(
        source_type=source_type,
        title=title,
        author=author,
        publisher=publisher,
        license=license,
        document_hash=document_hash,
    )
    db.add(source)
    db.flush()

    source_document = SourceDocument(
        academic_source_id=source.id,
        mime_type=mime_type,
        page_count=None,
        ingestion_status="processing",
    )
    db.add(source_document)
    db.flush()

    pages = extract_text(file_bytes, mime_type)
    source_document.page_count = len({p.page for p in pages if p.page is not None}) or None

    all_chunks = []
    for extracted_page in pages:
        for chunk in semantic_chunk(extracted_page.text, page=extracted_page.page):
            all_chunks.append(chunk)

    if not all_chunks:
        source_document.ingestion_status = "empty"
        db.flush()
        return source

    vectors = embedding_provider.embed_texts([c.text for c in all_chunks])

    chunk_rows = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        row = DocumentChunk(
            source_document_id=source_document.id,
            external_subject_id=external_subject_id,
            external_topic_id=external_topic_id,
            chunk_index=i,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            embedding_vector_id="",  # set below once we have the DB id
            token_count=chunk.word_count,
        )
        db.add(row)
        chunk_rows.append((row, vector))

    db.flush()  # assign chunk row ids

    points = []
    for row, vector in chunk_rows:
        row.embedding_vector_id = str(row.id)
        points.append(
            ChunkPoint(
                document_chunk_id=row.id,
                vector=vector,
                external_subject_id=external_subject_id,
                external_topic_id=external_topic_id,
                source_document_id=source_document.id,
                page=row.page,
                section=row.section,
                text=row.text,
            )
        )

    vector_store.upsert(points)
    source_document.ingestion_status = "ready"
    db.flush()
    return source
