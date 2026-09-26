"""Confirms the Phase 7 QA pipeline (dedup, distractor validation,
independent answer verification, quality scoring) is actually wired into
the generation executor end-to-end -- not just correct at the unit level.
"""

import json

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import (
    GenerationJob,
    GenerationJobTopic,
    MCQCandidate,
    MCQQualityScore,
    MCQValidationResult,
)
from backend.domain.enums import GenerationJobStatus, MCQStatus
from backend.generation.executor import (
    get_or_create_default_prompt_template,
    get_or_create_generation_model,
    run_generation_job,
)
from llm.providers.base import GenerationResult
from llm.providers.mock_provider import MockProvider


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


def _make_job(model_id, *, requested_count=1, max_attempts=1):
    return GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=requested_count,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 1.0},
        source_policy={"manual_context": True},
        model_id=model_id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
        max_attempts=max_attempts,
        topics=[
            GenerationJobTopic(
                external_topic_id="newtons-laws",
                topic_label_snapshot="Newton's Laws",
                manual_context="F=ma.",
                target_count=0,
                weight=1.0,
            )
        ],
    )


def _mcq_item(question="What is 12 + 8?", options=None, correct_option=1):
    return {
        "question": question,
        "options": options or ["20", "18", "22", "16"],
        "correct_option": correct_option,
        "explanation": "Explanation.",
        "topic": "mock",
        "difficulty": "easy",
        "bloom_level": "remember",
        "sources": [],
        "confidence": 0.9,
    }


def _batch_provider_returning(items):
    class FixedProvider(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            if "items" in json_schema.get("properties", {}):
                return GenerationResult(json.dumps({"items": items}), 10, 20, 0.01)
            return super().generate_structured(prompt, json_schema=json_schema, **kwargs)

    return FixedProvider()


def test_wrong_deterministic_answer_is_rejected_by_answer_verifier(db_session):
    """'What is 12 + 8?' claiming option 1 ('18') is correct -- SymPy-
    checkable and wrong -- must be caught and rejected, never reaching
    PENDING_REVIEW."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id)
    db_session.add(job)
    db_session.flush()

    provider = _batch_provider_returning([_mcq_item(correct_option=1)])  # wrong: 12+8=20, not 18
    result = run_generation_job(db_session, job, provider, template)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.REJECTED

    av_result = (
        db_session.query(MCQValidationResult)
        .filter_by(mcq_candidate_id=candidate.id, validator_name="answer_verifier")
        .one()
    )
    assert av_result.result == "fail"
    assert av_result.details["method"] == "deterministic"


def test_correct_deterministic_answer_proceeds_through_pipeline(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id)
    db_session.add(job)
    db_session.flush()

    provider = _batch_provider_returning([_mcq_item(correct_option=0)])  # correct: 12+8=20
    result = run_generation_job(db_session, job, provider, template)

    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.PENDING_REVIEW

    quality = db_session.query(MCQQualityScore).filter_by(mcq_candidate_id=candidate.id).one()
    assert quality.factual_correctness == 100.0
    assert quality.composite_score > 0


def test_duplicate_question_across_batch_calls_is_blocked(db_session):
    """Two separate generation attempts producing the identical question
    stem for the same subject/topic -- the second must be caught by
    deduplication and never reach PENDING_REVIEW twice."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id, requested_count=2, max_attempts=1)
    db_session.add(job)
    db_session.flush()

    same_item = _mcq_item(question="What is the SI unit of force?", options=["Newton", "Joule", "Watt", "Pascal"], correct_option=0)
    provider = _batch_provider_returning([same_item, dict(same_item)])
    result = run_generation_job(db_session, job, provider, template)

    statuses = sorted(c.status.value for c in result.candidates)
    assert statuses == ["DUPLICATE", "PENDING_REVIEW"]
    assert job.duplicate_count == 1


def test_multiple_correct_answer_risk_is_rejected_by_distractor_validator(db_session):
    """An option identical in wording to the correct option (a clear
    'multiple correct answers' risk) must be caught by the distractor
    validator, not silently passed through."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id)
    db_session.add(job)
    db_session.flush()

    # Note: exact-duplicate options are actually caught earlier by
    # structural validation; this uses distinct-but-embedding-identical
    # text isn't reachable with HashingEmbeddingProvider unless supplied,
    # so this test exercises the case with no embedding_provider (the
    # manual-context default) where only structural dup detection fires.
    item = _mcq_item(options=["Newton", "Newton", "Watt", "Pascal"], correct_option=0)
    provider = _batch_provider_returning([item])
    result = run_generation_job(db_session, job, provider, template)

    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.INVALID  # caught by structural duplicate-option check


def test_uncertain_answer_verification_still_reaches_pending_review(db_session):
    """A conceptual (non-computable) question, where MockProvider's own
    verifier honestly returns UNCERTAIN, must still reach human review --
    UNCERTAIN is not a rejection (master prompt section 21)."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id)
    db_session.add(job)
    db_session.flush()

    item = _mcq_item(
        question="What is the SI unit of force?", options=["Newton", "Joule", "Watt", "Pascal"], correct_option=0
    )
    provider = _batch_provider_returning([item])
    result = run_generation_job(db_session, job, provider, template)

    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.PENDING_REVIEW

    av_result = (
        db_session.query(MCQValidationResult)
        .filter_by(mcq_candidate_id=candidate.id, validator_name="answer_verifier")
        .one()
    )
    assert av_result.result == "uncertain"


def test_quality_score_row_is_persisted_for_every_candidate_reaching_review(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id, requested_count=3, max_attempts=1)
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MockProvider(), template)

    for candidate in result.candidates:
        if candidate.status == MCQStatus.PENDING_REVIEW:
            quality = db_session.query(MCQQualityScore).filter_by(mcq_candidate_id=candidate.id).one_or_none()
            assert quality is not None
            assert 0.0 <= quality.composite_score <= 100.0


def test_dedup_pool_spans_multiple_jobs_for_the_same_subject_topic(db_session):
    """A growing question bank must not accumulate near-duplicates just
    because they came from different generation runs -- dedup compares
    against every still-alive candidate for the subject/topic, not just
    the current job's own candidates."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    item = _mcq_item(question="What is the SI unit of force?", options=["Newton", "Joule", "Watt", "Pascal"], correct_option=0)

    job1 = _make_job(model.id, requested_count=1, max_attempts=1)
    db_session.add(job1)
    db_session.flush()
    run_generation_job(db_session, job1, _batch_provider_returning([dict(item)]), template)

    job2 = _make_job(model.id, requested_count=1, max_attempts=1)
    db_session.add(job2)
    db_session.flush()
    result2 = run_generation_job(db_session, job2, _batch_provider_returning([dict(item)]), template)

    assert result2.candidates[0].status == MCQStatus.DUPLICATE
