import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import GenerationJob, GenerationJobTopic
from backend.domain.enums import GenerationJobStatus, MCQStatus
from backend.generation.executor import (
    get_or_create_default_prompt_template,
    get_or_create_generation_model,
    run_generation_job,
)
from llm.providers.mock_provider import MockProvider


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


def test_run_generation_job_creates_pending_review_candidates(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock-model", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=6,
        option_count=4,
        difficulty_distribution={"easy": 0.5, "medium": 0.5},
        bloom_distribution={"remember": 0.5, "understand": 0.5},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        topics=[
            GenerationJobTopic(
                external_topic_id="newtons-laws",
                topic_label_snapshot="Newton's Laws",
                manual_context="F=ma. Force equals mass times acceleration.",
                target_count=0,
                weight=1.0,
            )
        ],
    )
    db_session.add(job)
    db_session.flush()

    provider = MockProvider("mock-model")
    result = run_generation_job(db_session, job, provider, template)

    assert result.status == GenerationJobStatus.COMPLETED
    assert result.generated_count == 6
    assert result.completed_at is not None

    candidates = result.candidates
    assert len(candidates) == 6
    assert all(c.status == MCQStatus.PENDING_REVIEW for c in candidates)
    for c in candidates:
        assert len(c.options) == 4
        assert sum(1 for o in c.options if o.is_correct) == 1


def test_run_generation_job_marks_invalid_candidates_on_bad_output(db_session, monkeypatch):
    model = get_or_create_generation_model(
        db_session, name="broken-model", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=2,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        topics=[
            GenerationJobTopic(
                external_topic_id="vectors",
                topic_label_snapshot="Vectors",
                manual_context="Vectors have magnitude and direction.",
                target_count=0,
                weight=1.0,
            )
        ],
    )
    db_session.add(job)
    db_session.flush()

    class BrokenProvider(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            from llm.providers.base import GenerationResult

            return GenerationResult("not valid json at all", 5, 5, 0.01)

    result = run_generation_job(db_session, job, BrokenProvider(), template)

    assert result.generated_count == 2
    assert result.rejected_count == 2
    assert all(c.status == MCQStatus.INVALID for c in result.candidates)
