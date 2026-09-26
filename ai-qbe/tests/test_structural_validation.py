import json
from pathlib import Path

from backend.validation.structural import validate_mcq_structure

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO_ROOT / "llm" / "schemas" / "mcq_schema.json").read_text())

VALID = {
    "question": "What is the SI unit of force?",
    "options": ["Joule", "Newton", "Watt", "Pascal"],
    "correct_option": 1,
    "explanation": "Force is measured in Newtons.",
    "topic": "Units",
    "difficulty": "easy",
    "bloom_level": "remember",
    "sources": [],
    "confidence": 0.9,
}


def test_valid_mcq_passes():
    result = validate_mcq_structure(VALID, SCHEMA)
    assert result.passed
    assert result.reasons == []


def test_duplicate_options_fail():
    broken = dict(VALID, options=["Newton", "newton", "Watt", "Pascal"])
    result = validate_mcq_structure(broken, SCHEMA)
    assert not result.passed
    assert "duplicate_options" in result.reasons


def test_correct_option_out_of_range_fails():
    broken = dict(VALID, correct_option=9)
    result = validate_mcq_structure(broken, SCHEMA)
    assert not result.passed
    assert "correct_option_out_of_range" in result.reasons


def test_overlong_option_flagged():
    broken = dict(VALID, options=["Newton", "a very long winded option exceeding six words total", "Watt", "Pascal"])
    result = validate_mcq_structure(broken, SCHEMA)
    assert not result.passed
    assert any("options_exceed_6_words" in r for r in result.reasons)


def test_short_stem_flagged():
    # Long enough to satisfy the JSON Schema's minLength=8, but still too
    # short to be a usable question stem -- exercises the stricter
    # length check, not the schema-level one.
    broken = dict(VALID, question="Why now?")
    result = validate_mcq_structure(broken, SCHEMA)
    assert not result.passed
    assert "question_stem_too_short" in result.reasons


def test_schema_violation_missing_field():
    broken = dict(VALID)
    del broken["bloom_level"]
    result = validate_mcq_structure(broken, SCHEMA)
    assert not result.passed
    assert result.reasons[0].startswith("schema_violation")
