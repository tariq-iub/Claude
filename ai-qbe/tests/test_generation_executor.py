import math

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import GenerationJob, GenerationJobTopic, GenerationMetric
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


def _make_job(model_id, *, requested_count, over_generation_factor=1.0, max_attempts=3):
    # Single-key difficulty/bloom/question-type distributions keep the
    # planner's output to exactly one cell, so the assertions below can
    # reason about an exact number of batch calls/candidates per round
    # rather than depending on largest-remainder rounding across a
    # multi-way split.
    return GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=requested_count,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        source_policy={"manual_context": True},
        model_id=model_id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=over_generation_factor,
        max_attempts=max_attempts,
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


def test_run_generation_job_creates_pending_review_candidates(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock-model", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    # over_generation_factor=1.0 keeps this test's expected count exactly
    # requested_count -- over-generation behavior itself is covered by
    # test_over_generation_factor_requests_more_than_requested_count below.
    job = _make_job(model.id, requested_count=6, over_generation_factor=1.0)
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


def test_over_generation_factor_requests_more_than_requested_count(db_session):
    """docs/PHASE0-DESIGN.md section 30: requesting exactly `requested_count`
    candidates would under-deliver against the human review queue once
    real rejection rates exist. With a 100%-valid MockProvider, round 0
    alone should already produce ceil(requested_count * over_generation_factor)
    candidates and stop there (no shortfall to chase)."""
    model = get_or_create_generation_model(
        db_session, name="mock-model", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = _make_job(model.id, requested_count=6, over_generation_factor=1.4)
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MockProvider("mock-model"), template)

    expected = math.ceil(6 * 1.4)
    assert result.generated_count == expected
    assert len(result.candidates) == expected
    assert all(c.status == MCQStatus.PENDING_REVIEW for c in result.candidates)

    metric_names = {m.metric_name: m.metric_value for m in db_session.query(GenerationMetric).filter_by(generation_job_id=result.id)}
    assert metric_names.get("target_met") == 1.0
    assert metric_names.get("rounds_used") == 1


def test_run_generation_job_marks_invalid_candidates_and_escalates_rounds(db_session):
    """A provider that never produces usable output should exhaust
    max_attempts rounds (over-generation escalating each round in an
    attempt to close the shortfall) rather than looping forever or
    silently giving up after one try."""
    model = get_or_create_generation_model(
        db_session, name="broken-model", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = _make_job(model.id, requested_count=2, over_generation_factor=1.0, max_attempts=3)
    db_session.add(job)
    db_session.flush()

    class BrokenProvider(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            from llm.providers.base import GenerationResult

            return GenerationResult("not valid json at all", 5, 5, 0.01)

    result = run_generation_job(db_session, job, BrokenProvider(), template)

    # Exactly one failed-batch candidate is recorded per round (this test's
    # single topic/difficulty/bloom/question-type cell always fits in one
    # batch call), and the job stops at max_attempts=3 rounds rather than
    # looping indefinitely chasing an unreachable target.
    assert result.generated_count == 3
    assert result.rejected_count == 3
    assert all(c.status == MCQStatus.INVALID for c in result.candidates)

    metrics = {
        m.metric_name: m.metric_value
        for m in db_session.query(GenerationMetric).filter_by(generation_job_id=result.id)
    }
    assert metrics.get("rounds_used") == 3
    assert metrics.get("target_met") == 0.0
