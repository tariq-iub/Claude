import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import AcademicSource, DocumentChunk, SourceDocument
from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from rag.ingestion import ingest_document
from rag.vectorstore import QdrantVectorStore


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
    return HashingEmbeddingProvider(dimension=64)


@pytest.fixture
def store():
    return QdrantVectorStore(location=":memory:", collection_name="ingestion_test")


def _make_pdf_bytes() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(
        0,
        10,
        "Newton's Second Law\n\n"
        "Force equals mass times acceleration. This relationship, F=ma, is one "
        "of the foundational equations of classical mechanics and explains how "
        "an object's velocity changes under an external force.",
    )
    return bytes(pdf.output())


def test_ingest_plain_text_document_creates_chunks_and_vectors(db_session, embedder, store):
    text = (
        b"Newton's Second Law\n\n"
        b"Force equals mass times acceleration. F=ma is a foundational law of "
        b"classical mechanics describing how force, mass, and acceleration relate."
    )
    source = ingest_document(
        db_session,
        file_bytes=text,
        mime_type="text/plain",
        title="Physics Notes",
        external_subject_id="phys-1",
        external_topic_id="newtons-laws",
        embedding_provider=embedder,
        vector_store=store,
    )

    chunks = db_session.query(DocumentChunk).all()
    assert len(chunks) >= 1
    assert chunks[0].external_subject_id == "phys-1"
    assert chunks[0].external_topic_id == "newtons-laws"
    assert chunks[0].embedding_vector_id  # populated after DB flush

    source_doc = db_session.query(SourceDocument).filter_by(academic_source_id=source.id).one()
    assert source_doc.ingestion_status == "ready"

    # Chunks are actually queryable from the vector store, not just the DB
    results = store.query(embedder.embed_query("F=ma force mass"), external_subject_id="phys-1", top_k=5)
    assert len(results) >= 1


def test_ingest_pdf_extracts_real_text(db_session, embedder, store):
    pdf_bytes = _make_pdf_bytes()
    ingest_document(
        db_session,
        file_bytes=pdf_bytes,
        mime_type="application/pdf",
        title="Physics Slides",
        external_subject_id="phys-1",
        external_topic_id="newtons-laws",
        embedding_provider=embedder,
        vector_store=store,
    )
    chunks = db_session.query(DocumentChunk).all()
    assert len(chunks) >= 1
    assert "Newton" in chunks[0].text
    assert chunks[0].page == 1


def test_ingest_same_bytes_twice_is_deduplicated(db_session, embedder, store):
    text = b"Some short but adequately long piece of academic content about vectors."
    first = ingest_document(
        db_session,
        file_bytes=text,
        mime_type="text/plain",
        title="Vectors Notes",
        external_subject_id="phys-1",
        external_topic_id="vectors",
        embedding_provider=embedder,
        vector_store=store,
    )
    second = ingest_document(
        db_session,
        file_bytes=text,
        mime_type="text/plain",
        title="Vectors Notes (re-upload)",
        external_subject_id="phys-1",
        external_topic_id="vectors",
        embedding_provider=embedder,
        vector_store=store,
    )
    assert first.id == second.id
    assert db_session.query(AcademicSource).count() == 1


def test_ingest_empty_document_marks_status_empty(db_session, embedder, store):
    source = ingest_document(
        db_session,
        file_bytes=b"   \n\n  ",
        mime_type="text/plain",
        title="Blank",
        external_subject_id="phys-1",
        external_topic_id="vectors",
        embedding_provider=embedder,
        vector_store=store,
    )
    source_doc = db_session.query(SourceDocument).filter_by(academic_source_id=source.id).one()
    assert source_doc.ingestion_status == "empty"
    assert db_session.query(DocumentChunk).count() == 0
