"""Tests Phase 5's real batch generation: one LLM call producing multiple
MCQs (docs/PHASE0-DESIGN.md section 11: 10-30 questions per call), not
one call per question.
"""

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import GenerationJob, GenerationJobTopic, MCQCandidate
from backend.domain.enums import GenerationJobStatus, MCQStatus
from backend.generation.executor import (
    build_batch_schema,
    get_or_create_default_prompt_template,
    get_or_create_generation_model,
    run_generation_job,
)
from backend.generation.planner import PlanCell
from llm.providers.mock_provider import MockProvider

MCQ_ITEM_SCHEMA = {
    "type": "object",
    "required": ["question", "options", "correct_option"],
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array"},
        "correct_option": {"type": "integer"},
    },
}


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


class CountingProvider(MockProvider):
    """Wraps MockProvider to count how many generate_structured calls
    happen -- the whole point of batching is fewer LLM calls for the same
    number of questions."""

    def __init__(self):
        super().__init__("counting-mock")
        self.call_count = 0

    def generate_structured(self, prompt, *, json_schema, **kwargs):
        self.call_count += 1
        return super().generate_structured(prompt, json_schema=json_schema, **kwargs)


def test_build_batch_schema_wraps_item_schema_in_items_array():
    schema = build_batch_schema(MCQ_ITEM_SCHEMA, max_items=20)
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["items"]["maxItems"] == 20
    assert schema["properties"]["items"]["items"] == MCQ_ITEM_SCHEMA


def test_run_generation_job_uses_one_call_per_batch_not_per_question(db_session):
    """A single cell needing 15 questions (under the default max batch
    size of 20) should take exactly one LLM call, not 15."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=15,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 1.0},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
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

    provider = CountingProvider()
    result = run_generation_job(db_session, job, provider, template)

    assert result.generated_count == 15
    assert provider.call_count == 1  # one batch call, not 15 single-item calls


def test_batch_exceeding_max_batch_size_splits_into_multiple_calls(db_session, monkeypatch):
    """A cell needing more than settings.generation_max_batch_size items
    must split across multiple calls, each capped at the configured max."""
    from backend.config import settings

    monkeypatch.setattr(settings, "generation_max_batch_size", 10)

    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=25,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 1.0},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
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
    db_session.add(job)
    db_session.flush()

    provider = CountingProvider()
    result = run_generation_job(db_session, job, provider, template)

    assert result.generated_count == 25
    assert provider.call_count == 3  # 10 + 10 + 5


def test_batch_items_are_distinct_not_identical_copies(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=8,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 1.0},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
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
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MockProvider(), template)
    stems = [c.question_stem for c in result.candidates]
    assert len(set(stems)) == len(stems)  # all distinct


def test_candidates_record_the_plan_cells_question_type(db_session):
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=20,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 0.5, "negative": 0.5},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
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
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MockProvider(), template)
    question_types = {c.question_type for c in result.candidates}
    assert question_types == {"single_best_answer", "negative"}


def test_model_returning_fewer_items_than_requested_is_not_a_batch_failure(db_session):
    """A model honestly returning fewer items than asked (e.g. it judged
    the context couldn't support more distinct questions) should not be
    treated as an error -- master prompt section 9's "if sufficient
    evidence is unavailable, do not generate the question" applies at the
    per-item level, not as an all-or-nothing batch contract."""
    model = get_or_create_generation_model(
        db_session, name="stingy-mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)

    class StingyProvider(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            result = super().generate_structured(prompt, json_schema=json_schema, **kwargs)
            if "items" in json_schema.get("properties", {}):
                import json as _json

                obj = _json.loads(result.text)
                obj["items"] = obj["items"][:2]  # only ever return 2, regardless of what was asked
                from llm.providers.base import GenerationResult

                return GenerationResult(_json.dumps(obj), result.prompt_tokens, result.completion_tokens, result.latency_seconds)
            return result

    job = GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=5,
        option_count=4,
        difficulty_distribution={"easy": 1.0},
        bloom_distribution={"remember": 1.0},
        question_type_distribution={"single_best_answer": 1.0},
        source_policy={"manual_context": True},
        model_id=model.id,
        status=GenerationJobStatus.QUEUED,
        created_by="tester",
        over_generation_factor=1.0,
        max_attempts=3,
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
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, StingyProvider(), template)
    valid_candidates = [c for c in result.candidates if c.status == MCQStatus.PENDING_REVIEW]
    # Never crashes, and the over-generation round loop keeps trying
    # (bounded by max_attempts) rather than giving up after the first
    # under-sized batch.
    assert len(valid_candidates) > 0
    assert result.status == GenerationJobStatus.COMPLETED
