import json

from backend.validation.answer_verification import verify_answer
from llm.providers.base import GenerationResult
from llm.providers.mock_provider import MockProvider


def test_deterministic_path_used_for_solvable_arithmetic():
    result = verify_answer(
        MockProvider(), question_stem="What is 12 + 8?", options=["20", "18", "22", "16"], correct_option=0
    )
    assert result.method == "deterministic"
    assert result.verdict == "PASS"


def test_deterministic_path_fails_wrong_claim():
    result = verify_answer(
        MockProvider(), question_stem="What is 12 + 8?", options=["20", "18", "22", "16"], correct_option=1
    )
    assert result.method == "deterministic"
    assert result.verdict == "FAIL"


def test_falls_back_to_llm_for_conceptual_question():
    result = verify_answer(
        MockProvider(),
        question_stem="What is the SI unit of force?",
        options=["Newton", "Joule", "Watt", "Pascal"],
        correct_option=0,
    )
    assert result.method == "llm"
    assert result.verdict == "UNCERTAIN"  # MockProvider's honest non-answer, per Phase 1 design


class VerifierAgreesProvider(MockProvider):
    def generate_structured(self, prompt, *, json_schema, **kwargs):
        if "verdict" in json_schema.get("properties", {}):
            obj = {"verdict": "PASS", "confidence": 0.95, "reason": "Matches known physics.", "evidence": []}
            return GenerationResult(json.dumps(obj), 10, 10, 0.01)
        return super().generate_structured(prompt, json_schema=json_schema, **kwargs)


class VerifierDisagreesProvider(MockProvider):
    def generate_structured(self, prompt, *, json_schema, **kwargs):
        if "verdict" in json_schema.get("properties", {}):
            obj = {
                "verdict": "FAIL",
                "confidence": 0.9,
                "reason": "The claimed answer is incorrect.",
                "derived_correct_option": 2,
                "evidence": [],
            }
            return GenerationResult(json.dumps(obj), 10, 10, 0.01)
        return super().generate_structured(prompt, json_schema=json_schema, **kwargs)


def test_llm_verifier_pass_is_respected():
    result = verify_answer(
        VerifierAgreesProvider(),
        question_stem="What is the SI unit of force?",
        options=["Newton", "Joule", "Watt", "Pascal"],
        correct_option=0,
    )
    assert result.verdict == "PASS"
    assert result.method == "llm"


def test_llm_verifier_fail_is_respected_and_never_trusted_as_pass():
    result = verify_answer(
        VerifierDisagreesProvider(),
        question_stem="What is the SI unit of force?",
        options=["Newton", "Joule", "Watt", "Pascal"],
        correct_option=0,
    )
    assert result.verdict == "FAIL"
    assert result.derived_correct_option == 2


def test_unparseable_llm_response_becomes_uncertain_not_pass():
    class BrokenVerifier(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            return GenerationResult("not json", 5, 5, 0.01)

    result = verify_answer(
        BrokenVerifier(),
        question_stem="What is the SI unit of force?",
        options=["Newton", "Joule", "Watt", "Pascal"],
        correct_option=0,
    )
    assert result.verdict == "UNCERTAIN"


def test_context_is_passed_through_to_verifier_prompt():
    captured = {}

    class CapturingProvider(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            captured["prompt"] = prompt
            return super().generate_structured(prompt, json_schema=json_schema, **kwargs)

    verify_answer(
        CapturingProvider(),
        question_stem="What is the SI unit of force?",
        options=["Newton", "Joule", "Watt", "Pascal"],
        correct_option=0,
        context="Newton's laws define force in terms of mass and acceleration.",
    )
    assert "Newton's laws define force" in captured["prompt"]
