"""Tests QdrantVectorStore against an embedded (`:memory:`) Qdrant
instance -- the one RAG component that runs against real Qdrant code
(not a mock) in this sandbox, since qdrant-client's embedded mode needs
no server or network access.
"""

import pytest

from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from rag.vectorstore import ChunkPoint, QdrantVectorStore


@pytest.fixture
def store():
    return QdrantVectorStore(location=":memory:", collection_name="test_collection")


@pytest.fixture
def embedder():
    return HashingEmbeddingProvider(dimension=64)


def _point(chunk_id, text, subject="phys-1", topic="newtons-laws", embedder=None):
    return ChunkPoint(
        document_chunk_id=chunk_id,
        vector=embedder.embed_query(text),
        external_subject_id=subject,
        external_topic_id=topic,
        source_document_id=1,
        page=1,
        section="Second Law",
        text=text,
    )


def test_query_on_empty_collection_returns_no_results(store, embedder):
    results = store.query(embedder.embed_query("anything"), external_subject_id="phys-1")
    assert results == []


def test_upsert_and_query_returns_the_upserted_chunk(store, embedder):
    point = _point(1, "Force equals mass times acceleration", embedder=embedder)
    store.upsert([point])

    results = store.query(embedder.embed_query("mass acceleration force"), external_subject_id="phys-1")
    assert len(results) == 1
    assert results[0].document_chunk_id == 1
    assert results[0].section == "Second Law"


def test_query_filters_by_subject(store, embedder):
    store.upsert([_point(1, "Newton's second law: F=ma", subject="phys-1", embedder=embedder)])
    store.upsert([_point(2, "Balancing chemical equations", subject="chem-1", topic="stoichiometry", embedder=embedder)])

    results = store.query(embedder.embed_query("F=ma"), external_subject_id="phys-1")
    assert {r.document_chunk_id for r in results} == {1}

    results = store.query(embedder.embed_query("F=ma"), external_subject_id="chem-1")
    assert {r.document_chunk_id for r in results} == {2}


def test_query_filters_by_topic_within_subject(store, embedder):
    store.upsert([_point(1, "Newton's second law content", subject="phys-1", topic="newtons-laws", embedder=embedder)])
    store.upsert([_point(2, "Vector addition content", subject="phys-1", topic="vectors", embedder=embedder)])

    results = store.query(
        embedder.embed_query("content"), external_subject_id="phys-1", external_topic_id="newtons-laws"
    )
    assert {r.document_chunk_id for r in results} == {1}


def test_upsert_same_chunk_id_overwrites_not_duplicates(store, embedder):
    store.upsert([_point(1, "Original text", embedder=embedder)])
    store.upsert([_point(1, "Updated text", embedder=embedder)])

    results = store.query(embedder.embed_query("Updated text"), external_subject_id="phys-1", top_k=10)
    assert len(results) == 1
    assert results[0].text == "Updated text"


def test_delete_by_source_document_removes_only_its_chunks(store, embedder):
    store.upsert(
        [
            ChunkPoint(
                document_chunk_id=1,
                vector=embedder.embed_query("doc one text"),
                external_subject_id="phys-1",
                external_topic_id="newtons-laws",
                source_document_id=100,
                page=1,
                section=None,
                text="doc one text",
            ),
            ChunkPoint(
                document_chunk_id=2,
                vector=embedder.embed_query("doc two text"),
                external_subject_id="phys-1",
                external_topic_id="newtons-laws",
                source_document_id=200,
                page=1,
                section=None,
                text="doc two text",
            ),
        ]
    )
    store.delete_by_source_document(100)
    results = store.query(embedder.embed_query("text"), external_subject_id="phys-1", top_k=10)
    assert {r.document_chunk_id for r in results} == {2}


def test_top_k_limits_result_count(store, embedder):
    for i in range(10):
        store.upsert([_point(i, f"Newton's law content variant {i}", embedder=embedder)])
    results = store.query(embedder.embed_query("Newton's law content"), external_subject_id="phys-1", top_k=3)
    assert len(results) == 3
