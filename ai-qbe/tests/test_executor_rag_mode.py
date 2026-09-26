"""Confirms the generation executor actually uses RAG-retrieved context
(not manual_context) when a job's source_policy enables it, records
MCQSource provenance for the chunks used, and falls back to
manual_context when RAG finds no evidence for a topic.
"""

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import GenerationJob, GenerationJobTopic, MCQSource
from backend.domain.enums import GenerationJobStatus
from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from backend.generation.executor import (
    get_or_create_default_prompt_template,
    get_or_create_generation_model,
    resolve_topic_context,
    run_generation_job,
)
from llm.providers.mock_provider import MockProvider
from rag.vectorstore import ChunkPoint, QdrantVectorStore


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def embedder():
    return HashingEmbeddingProvider(dimension=128)


@pytest.fixture
def store():
    return QdrantVectorStore(location=":memory:", collection_name="executor_rag_test")


def _seed_chunks(store, embedder):
    store.upsert(
        [
            ChunkPoint(
                document_chunk_id=1,
                vector=embedder.embed_query("Newton's second law: F=ma relates force, mass, acceleration."),
                external_subject_id="phys-1",
                external_topic_id="newtons-laws",
                source_document_id=1,
                page=1,
                section="Second Law",
                text="Newton's second law: F=ma relates force, mass, acceleration.",
            )
        ]
    )


def _make_job(db_session, model_id, *, use_rag, manual_context=None):
    return GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=2,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        source_policy={"local_docs": use_rag, "manual_context": not use_rag},
        model_id=model_id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        topics=[
            GenerationJobTopic(
                external_topic_id="newtons-laws",
                topic_label_snapshot="Newton's Laws",
                manual_context=manual_context,
                target_count=0,
                weight=1.0,
            )
        ],
    )


def test_resolve_topic_context_uses_rag_when_evidence_exists(db_session, embedder, store):
    _seed_chunks(store, embedder)
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    job = _make_job(db_session, model.id, use_rag=True)
    db_session.add(job)
    db_session.flush()

    context, chunk_ids = resolve_topic_context(
        job, job.topics[0], vector_store=store, embedding_provider=embedder
    )
    assert "F=ma" in context
    assert chunk_ids == [1]


def test_resolve_topic_context_falls_back_to_manual_when_no_evidence(db_session, embedder, store):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    job = _make_job(db_session, model.id, use_rag=True, manual_context="Fallback context: vectors have direction.")
    db_session.add(job)
    db_session.flush()

    # No chunks ingested for this topic in `store` -- RAG finds nothing.
    context, chunk_ids = resolve_topic_context(
        job, job.topics[0], vector_store=store, embedding_provider=embedder
    )
    assert context == "Fallback context: vectors have direction."
    assert chunk_ids == []


def test_manual_context_mode_never_touches_vector_store(db_session, embedder, store):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    job = _make_job(db_session, model.id, use_rag=False, manual_context="Manually supplied context only.")
    db_session.add(job)
    db_session.flush()

    context, chunk_ids = resolve_topic_context(
        job, job.topics[0], vector_store=store, embedding_provider=embedder
    )
    assert context == "Manually supplied context only."
    assert chunk_ids == []


def test_run_generation_job_records_mcqsource_citations_in_rag_mode(db_session, embedder, store):
    _seed_chunks(store, embedder)
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(db_session, model.id, use_rag=True)
    db_session.add(job)
    db_session.flush()

    run_generation_job(db_session, job, MockProvider(), template, vector_store=store, embedding_provider=embedder)

    assert job.generated_count == 2
    for candidate in job.candidates:
        sources = db_session.query(MCQSource).filter_by(mcq_candidate_id=candidate.id).all()
        assert len(sources) == 1
        assert sources[0].document_chunk_id == 1
