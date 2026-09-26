"""Unit tests for scripts/benchmark/validators.py.

These run without a GPU or any LLM runtime -- they exercise the automated
grading logic itself against fixed inputs, which is exactly the part of
the Phase 1 harness that can be verified in an environment with no model
access. The LLM-facing parts (run_benchmark.py's provider calls) need the
real target workstation and are documented, not executed, here.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import validators  # noqa: E402

MCQ_SCHEMA = json.loads((REPO_ROOT / "llm" / "schemas" / "mcq_schema.json").read_text())
VERIFICATION_SCHEMA = json.loads(
    (REPO_ROOT / "llm" / "schemas" / "verification_schema.json").read_text()
)


def test_extract_json_plain():
    text = '{"a": 1, "b": 2}'
    assert validators.extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_fenced_markdown():
    text = 'Sure, here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert validators.extract_json(text) == {"a": 1}


def test_extract_json_prose_prefix_suffix():
    text = 'The answer is: {"a": 1} -- let me know if you need more.'
    assert validators.extract_json(text) == {"a": 1}


def test_extract_json_returns_none_on_garbage():
    assert validators.extract_json("not json at all") is None


VALID_MCQ = {
    "question": "What is the SI unit of force?",
    "options": ["Joule", "Newton", "Watt", "Pascal"],
    "correct_option": 1,
    "explanation": "Force is measured in Newtons per F=ma.",
    "topic": "Units and Measurements",
    "difficulty": "easy",
    "bloom_level": "remember",
    "sources": [],
    "confidence": 0.95,
}


def test_grade_structural_valid_mcq_passes():
    result = validators.grade_structural(json.dumps(VALID_MCQ), MCQ_SCHEMA)
    assert result.passed
    assert result.score == 1.0
    assert result.details["parsed"]["question"] == VALID_MCQ["question"]


def test_grade_structural_rejects_missing_required_field():
    broken = dict(VALID_MCQ)
    del broken["correct_option"]
    result = validators.grade_structural(json.dumps(broken), MCQ_SCHEMA)
    assert not result.passed
    assert result.details["reason"] == "schema_violation"


def test_grade_structural_rejects_extra_field():
    broken = dict(VALID_MCQ)
    broken["extra_field_not_in_schema"] = "nope"
    result = validators.grade_structural(json.dumps(broken), MCQ_SCHEMA)
    assert not result.passed


def test_grade_structural_rejects_unparseable_text():
    result = validators.grade_structural("this is not json", MCQ_SCHEMA)
    assert not result.passed
    assert result.details["reason"] == "not_json_parseable"


def test_grade_option_conciseness_all_short():
    result = validators.grade_option_conciseness(["Newton", "Joule", "Watt", "Pascal"])
    assert result.passed
    assert result.score == 1.0


def test_grade_option_conciseness_flags_long_option():
    options = ["Newton", "a very long option that exceeds six words by a lot", "Watt", "Pascal"]
    result = validators.grade_option_conciseness(options)
    assert not result.passed
    assert result.score < 1.0


def test_grade_distractor_hygiene_no_duplicates():
    result = validators.grade_distractor_hygiene(["Newton", "Joule", "Watt", "Pascal"])
    assert result.passed
    assert result.details["duplicate_count"] == 0


def test_grade_distractor_hygiene_detects_duplicate():
    result = validators.grade_distractor_hygiene(["Newton", "newton ", "Watt", "Pascal"])
    assert not result.passed
    assert result.details["duplicate_count"] == 1


def test_grade_single_correct_valid_index():
    result = validators.grade_single_correct(VALID_MCQ)
    assert result.passed


def test_grade_single_correct_out_of_range():
    broken = dict(VALID_MCQ)
    broken["correct_option"] = 99
    result = validators.grade_single_correct(broken)
    assert not result.passed


def test_grade_verification_accuracy_correct_pass():
    result = validators.grade_verification_accuracy({"verdict": "PASS"}, "PASS")
    assert result.passed
    assert result.score == 1.0


def test_grade_verification_accuracy_wrong_verdict_fails():
    result = validators.grade_verification_accuracy({"verdict": "PASS"}, "FAIL")
    assert not result.passed
    assert result.score == 0.0


def test_grade_verification_accuracy_uncertain_gets_partial_credit():
    result = validators.grade_verification_accuracy({"verdict": "UNCERTAIN"}, "FAIL")
    assert not result.passed
    assert result.score == 0.5


def test_verification_schema_accepts_valid_payload():
    import jsonschema

    payload = {"verdict": "PASS", "confidence": 0.9, "reason": "matches computed value"}
    jsonschema.validate(payload, VERIFICATION_SCHEMA)  # should not raise


def test_grade_math_answer_diff_correct():
    task = {
        "sympy_op": "diff",
        "sympy_var": "x",
        "expected_sympy": "4*x**3 + 6*x",
    }
    model_output = json.dumps({"answer": "4*x**3 + 6*x"})
    result = validators.grade_math_answer(model_output, task)
    assert result.passed


def test_grade_math_answer_diff_equivalent_but_differently_written():
    task = {"sympy_op": "diff", "sympy_var": "x", "expected_sympy": "4*x**3 + 6*x"}
    model_output = json.dumps({"answer": "6*x + 4*x**3"})  # reordered, still equal
    result = validators.grade_math_answer(model_output, task)
    assert result.passed


def test_grade_math_answer_wrong_value_fails():
    task = {"sympy_op": "diff", "sympy_var": "x", "expected_sympy": "4*x**3 + 6*x"}
    model_output = json.dumps({"answer": "4*x**3 + 5*x"})
    result = validators.grade_math_answer(model_output, task)
    assert not result.passed


def test_grade_math_answer_solve_matches_regardless_of_order():
    task = {"sympy_op": "solve", "expected_sympy": "[2, 3]"}
    model_output = json.dumps({"answer": "[3, 2]"})
    result = validators.grade_math_answer(model_output, task)
    assert result.passed


def test_grade_math_answer_linsolve_matches():
    task = {"sympy_op": "linsolve", "expected_sympy": "{(3, 2)}"}
    model_output = json.dumps({"answer": "{(3, 2)}"})
    result = validators.grade_math_answer(model_output, task)
    assert result.passed


def test_grade_math_answer_handles_garbage_gracefully():
    task = {"sympy_op": "diff", "sympy_var": "x", "expected_sympy": "4*x**3 + 6*x"}
    model_output = json.dumps({"answer": "not a valid expression $$$"})
    result = validators.grade_math_answer(model_output, task)
    assert not result.passed
    assert result.details["reason"] in ("parse_or_eval_error",)


def test_grade_math_answer_missing_answer_field():
    task = {"sympy_op": "diff", "sympy_var": "x", "expected_sympy": "4*x**3 + 6*x"}
    model_output = json.dumps({"not_answer": "whatever"})
    result = validators.grade_math_answer(model_output, task)
    assert not result.passed
    assert result.details["reason"] == "no_answer_field"
