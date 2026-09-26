from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from rag.knowledge_pack import build_topic_knowledge_pack
from rag.vectorstore import ChunkPoint, QdrantVectorStore


def _seed(store, embedder):
    chunks = [
        (1, "Newton's second law: force equals mass times acceleration (F=ma).", "Second Law", 12),
        (2, "Newton's third law: for every action there is an equal and opposite reaction.", "Third Law", 15),
        (3, "Vectors have both magnitude and direction, unlike scalars.", "Vectors", 3),
    ]
    points = [
        ChunkPoint(
            document_chunk_id=cid,
            vector=embedder.embed_query(text),
            external_subject_id="phys-1",
            external_topic_id="newtons-laws" if "Newton" in text else "vectors",
            source_document_id=1,
            page=page,
            section=section,
            text=text,
        )
        for cid, text, section, page in chunks
    ]
    store.upsert(points)


def test_knowledge_pack_retrieves_topic_scoped_evidence():
    embedder = HashingEmbeddingProvider(dimension=128)
    store = QdrantVectorStore(location=":memory:", collection_name="kp_test")
    _seed(store, embedder)

    pack = build_topic_knowledge_pack(
        store, embedder,
        external_subject_id="phys-1",
        external_topic_id="newtons-laws",
        topic_label="Newton's Laws",
    )
    assert pack.has_evidence
    ids = {c.document_chunk_id for c in pack.retrieved_chunks}
    assert ids.issubset({1, 2})
    assert 3 not in ids  # vectors chunk must not leak into newtons-laws topic


def test_knowledge_pack_with_no_ingested_documents_has_no_evidence():
    embedder = HashingEmbeddingProvider(dimension=128)
    store = QdrantVectorStore(location=":memory:", collection_name="kp_empty_test")

    pack = build_topic_knowledge_pack(
        store, embedder,
        external_subject_id="phys-1",
        external_topic_id="gravitation",
        topic_label="Gravitation",
    )
    assert not pack.has_evidence
    assert pack.as_context_text() == ""
    assert pack.source_chunk_ids == []


def test_knowledge_pack_context_text_includes_citations():
    embedder = HashingEmbeddingProvider(dimension=128)
    store = QdrantVectorStore(location=":memory:", collection_name="kp_citation_test")
    _seed(store, embedder)

    pack = build_topic_knowledge_pack(
        store, embedder,
        external_subject_id="phys-1",
        external_topic_id="newtons-laws",
        topic_label="Newton's Laws",
    )
    context = pack.as_context_text()
    assert "[chunk:1]" in context or "[chunk:2]" in context
    assert "section:" in context
