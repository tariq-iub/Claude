"""Confirms the generation executor actually wires notation validation
into the pipeline: a candidate with malformed LaTeX/chemistry never
reaches PENDING_REVIEW, and a real validation-result row records why.
"""

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.base import Base, make_engine
from backend.database.models import GenerationJob, GenerationJobTopic, MCQValidationResult
from backend.domain.enums import GenerationJobStatus, MCQStatus
from backend.generation.executor import (
    get_or_create_default_prompt_template,
    get_or_create_generation_model,
    run_generation_job,
)
from backend.validation.mathjax_bridge import is_available
from llm.providers.base import GenerationResult
from llm.providers.mock_provider import MockProvider

pytestmark = pytest.mark.skipif(
    not is_available(), reason="Node.js + tools/mathjax_validator's mathjax-full dependency not installed"
)


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    yield session
    session.close()


def _make_job(model_id, *, max_attempts=3):
    return GenerationJob(
        external_subject_id="phys-1",
        subject_label_snapshot="Physics-I",
        requested_count=1,
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


class MalformedLatexProvider(MockProvider):
    """Always returns a structurally-valid MCQ whose stem contains
    unrenderable LaTeX, to exercise the notation gate specifically
    (isolated from the structural validator, which would already catch
    a missing field)."""

    def generate_structured(self, prompt, *, json_schema, **kwargs):
        import json

        if "items" in json_schema.get("properties", {}):
            item = {
                "question": "What does $\\frac{a}{b$ represent?",  # unbalanced brace
                "options": ["A ratio", "A sum", "A product", "A limit"],
                "correct_option": 0,
                "explanation": "Explanation text.",
                "topic": "mock",
                "difficulty": "easy",
                "bloom_level": "remember",
                "sources": [],
                "confidence": 0.9,
            }
            return GenerationResult(json.dumps({"items": [item]}), 10, 20, 0.01)
        return super().generate_structured(prompt, json_schema=json_schema, **kwargs)


class UnbalancedChemistryProvider(MockProvider):
    def generate_structured(self, prompt, *, json_schema, **kwargs):
        import json

        if "items" in json_schema.get("properties", {}):
            item = {
                "question": "Which reaction is this: $\\ce{H2 + O2 -> H2O}$?",
                "options": ["Combustion", "Neutralization", "Decomposition", "None"],
                "correct_option": 0,
                "explanation": "Water forms from hydrogen and oxygen.",
                "topic": "mock",
                "difficulty": "easy",
                "bloom_level": "remember",
                "sources": [],
                "confidence": 0.9,
            }
            return GenerationResult(json.dumps({"items": [item]}), 10, 20, 0.01)
        return super().generate_structured(prompt, json_schema=json_schema, **kwargs)


def test_malformed_latex_candidate_is_marked_invalid(db_session):
    model = get_or_create_generation_model(
        db_session, name="malformed-latex-mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id, max_attempts=1)
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MalformedLatexProvider(), template)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.INVALID

    notation_results = (
        db_session.query(MCQValidationResult)
        .filter_by(mcq_candidate_id=candidate.id, validator_name="notation")
        .all()
    )
    assert len(notation_results) == 1
    assert notation_results[0].result == "fail"
    assert "malformed LaTeX" in str(notation_results[0].details["reasons"])


def test_unbalanced_chemical_equation_candidate_is_marked_invalid(db_session):
    model = get_or_create_generation_model(
        db_session, name="unbalanced-chem-mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id, max_attempts=1)
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, UnbalancedChemistryProvider(), template)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.status == MCQStatus.INVALID

    notation_results = (
        db_session.query(MCQValidationResult)
        .filter_by(mcq_candidate_id=candidate.id, validator_name="notation")
        .all()
    )
    assert notation_results[0].result == "fail"
    assert "unbalanced" in str(notation_results[0].details["reasons"]).lower()


def test_clean_mock_generated_candidates_pass_notation_and_reach_pending_review(db_session):
    """MockProvider's own default output contains no LaTeX at all, so it
    should sail through the notation gate untouched -- confirms Phase 6
    didn't regress the plain-text-only path exercised throughout earlier
    phases' tests."""
    model = get_or_create_generation_model(
        db_session, name="mock", provider_type="mock", version="v1", quantization=None, context_window=4096
    )
    template = get_or_create_default_prompt_template(db_session, model.id)
    job = _make_job(model.id)
    db_session.add(job)
    db_session.flush()

    result = run_generation_job(db_session, job, MockProvider(), template)

    assert len(result.candidates) == 1
    assert result.candidates[0].status == MCQStatus.PENDING_REVIEW
    notation_results = (
        db_session.query(MCQValidationResult)
        .filter_by(mcq_candidate_id=result.candidates[0].id, validator_name="notation")
        .all()
    )
    assert notation_results[0].result == "pass"
