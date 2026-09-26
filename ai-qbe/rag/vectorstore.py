"""IVectorStore: the vector-database abstraction (docs/PHASE0-DESIGN.md
section 7). Qdrant is the chosen implementation (self-hosted, single
binary, metadata filtering + ANN in one place); the interface exists so a
pgvector-backed implementation could be dropped in for an installation
that wants to minimize service count, without touching ingestion or
retrieval code.

`QdrantVectorStore` works identically against `location=":memory:"`
(an embedded instance, no server -- used for dev/test, including this
repository's own test suite) and a real `location="http://host:6333"`
server, since qdrant-client exposes the same API either way. That's what
makes this the one RAG component actually exercised for real (not
mocked) in this development sandbox, unlike the LLM and embedding model
paths.
"""

from __future__ import annotations

import abc
import dataclasses
import uuid


@dataclasses.dataclass
class ChunkPoint:
    """One chunk to upsert: its embedding vector plus the metadata needed
    for filtered retrieval (docs/PHASE0-DESIGN.md section 7's "semantic
    relevance + metadata filters").
    """

    document_chunk_id: int
    vector: list[float]
    external_subject_id: str
    external_topic_id: str | None
    source_document_id: int
    page: int | None
    section: str | None
    text: str


@dataclasses.dataclass
class ScoredChunk:
    document_chunk_id: int
    score: float
    text: str
    page: int | None
    section: str | None
    source_document_id: int


class IVectorStore(abc.ABC):
    @abc.abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        ...

    @abc.abstractmethod
    def upsert(self, points: list[ChunkPoint]) -> None:
        ...

    @abc.abstractmethod
    def query(
        self,
        vector: list[float],
        *,
        external_subject_id: str,
        external_topic_id: str | None = None,
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        ...

    @abc.abstractmethod
    def delete_by_source_document(self, source_document_id: int) -> None:
        ...


class QdrantVectorStore(IVectorStore):
    COLLECTION_NAME = "aiqbe_chunks"

    def __init__(self, location: str = ":memory:", collection_name: str | None = None):
        from qdrant_client import QdrantClient

        self._client = QdrantClient(location=location)
        self._collection_name = collection_name or self.COLLECTION_NAME
        self._ensured = False

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._ensured:
            return
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                self._collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        self._ensured = True

    def upsert(self, points: list[ChunkPoint]) -> None:
        from qdrant_client.models import PointStruct

        if not points:
            return
        self.ensure_collection(len(points[0].vector))
        qdrant_points = [
            PointStruct(
                # Qdrant point ids are independent of our own primary keys,
                # but deterministic from document_chunk_id so re-ingesting
                # the same chunk overwrites rather than duplicates it.
                id=_deterministic_point_id(p.document_chunk_id),
                vector=p.vector,
                payload={
                    "document_chunk_id": p.document_chunk_id,
                    "external_subject_id": p.external_subject_id,
                    "external_topic_id": p.external_topic_id,
                    "source_document_id": p.source_document_id,
                    "page": p.page,
                    "section": p.section,
                    "text": p.text,
                },
            )
            for p in points
        ]
        self._client.upsert(self._collection_name, points=qdrant_points)

    def query(
        self,
        vector: list[float],
        *,
        external_subject_id: str,
        external_topic_id: str | None = None,
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self._collection_name):
            return []

        must = [FieldCondition(key="external_subject_id", match=MatchValue(value=external_subject_id))]
        if external_topic_id is not None:
            must.append(FieldCondition(key="external_topic_id", match=MatchValue(value=external_topic_id)))

        result = self._client.query_points(
            self._collection_name,
            query=vector,
            query_filter=Filter(must=must),
            limit=top_k,
        )
        return [
            ScoredChunk(
                document_chunk_id=pt.payload["document_chunk_id"],
                score=pt.score,
                text=pt.payload["text"],
                page=pt.payload.get("page"),
                section=pt.payload.get("section"),
                source_document_id=pt.payload["source_document_id"],
            )
            for pt in result.points
        ]

    def delete_by_source_document(self, source_document_id: int) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        if not self._client.collection_exists(self._collection_name):
            return
        self._client.delete(
            self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source_document_id", match=MatchValue(value=source_document_id))]
                )
            ),
        )


def _deterministic_point_id(document_chunk_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"aiqbe-chunk-{document_chunk_id}"))
