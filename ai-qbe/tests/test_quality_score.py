from backend.validation.answer_verification import AnswerVerificationResult
from backend.validation.dedup import DedupResult
from backend.validation.difficulty_bloom import DifficultyBloomCheckResult
from backend.validation.distractor import DistractorValidationResult
from backend.validation.notation import NotationValidationResult
from backend.validation.quality_score import compute_quality_score
from backend.validation.structural import StructuralValidationResult


def _perfect_inputs():
    return dict(
        structural=StructuralValidationResult(passed=True),
        notation=NotationValidationResult(passed=True, mathjax_checked=True, had_math_content=False),
        answer_verification=AnswerVerificationResult(verdict="PASS", method="llm"),
        distractor=DistractorValidationResult(passed=True, embedding_checked=True, score=100.0),
        dedup=DedupResult(is_duplicate=False),
        difficulty_bloom=DifficultyBloomCheckResult(
            estimated_bloom=None, estimated_difficulty=None, bloom_match=True, difficulty_match=True
        ),
        has_sources=True,
        used_manual_context=False,
        had_any_context=True,
    )


def test_all_validators_passing_produces_high_composite_score():
    score = compute_quality_score(**_perfect_inputs())
    assert score.composite_score > 90.0
    assert score.factual_correctness == 100.0
    assert score.notation_validity == 100.0


def test_never_uses_llm_self_reported_confidence():
    """The generator's own `confidence` field must never appear anywhere
    in the scoring inputs -- quality is derived purely from validator
    outputs (master prompt section 25)."""
    import ast
    import inspect

    from backend.validation import quality_score as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    # Strip the module docstring (which explains the rule in prose, and
    # necessarily mentions the forbidden field names to do so) before
    # checking the actual code for a "confidence" attribute/subscript
    # access or a "raw_item" reference.
    if ast.get_docstring(tree) is not None:
        tree.body = tree.body[1:]
    code_only = ast.unparse(tree)
    assert "confidence" not in code_only
    assert "raw_item" not in code_only


def test_uncertain_answer_verification_reduces_factual_correctness_but_not_to_zero():
    inputs = _perfect_inputs()
    inputs["answer_verification"] = AnswerVerificationResult(verdict="UNCERTAIN", method="llm")
    score = compute_quality_score(**inputs)
    assert 0 < score.factual_correctness < 100
    assert score.composite_score < 100


def test_structural_failure_zeroes_clarity_and_single_correctness():
    inputs = _perfect_inputs()
    inputs["structural"] = StructuralValidationResult(passed=False, reasons=["duplicate_options"])
    score = compute_quality_score(**inputs)
    assert score.clarity == 0.0
    assert score.single_correctness == 0.0


def test_no_sources_and_no_context_gives_low_grounding_score():
    inputs = _perfect_inputs()
    inputs["has_sources"] = False
    inputs["used_manual_context"] = False
    inputs["had_any_context"] = False
    score = compute_quality_score(**inputs)
    assert score.source_grounding <= 20.0


def test_manual_context_without_citations_scores_between_ungrounded_and_cited():
    inputs = _perfect_inputs()
    inputs["has_sources"] = False
    inputs["used_manual_context"] = True
    inputs["had_any_context"] = True
    score = compute_quality_score(**inputs)
    assert 20.0 < score.source_grounding < 100.0


def test_unchecked_distractor_validation_caps_the_distractor_score():
    inputs = _perfect_inputs()
    inputs["distractor"] = DistractorValidationResult(passed=True, embedding_checked=False, score=100.0)
    score = compute_quality_score(**inputs)
    assert score.distractor_quality <= 75.0


def test_duplicate_risk_score_reflects_borderline_clearance():
    inputs = _perfect_inputs()
    inputs["dedup"] = DedupResult(is_duplicate=False, score=0.85)
    score = compute_quality_score(**inputs)
    assert score.duplicate_risk < 100.0


def test_ambiguous_difficulty_bloom_match_is_neutral_not_penalized_hard():
    inputs = _perfect_inputs()
    inputs["difficulty_bloom"] = DifficultyBloomCheckResult(
        estimated_bloom=None, estimated_difficulty=None, bloom_match=None, difficulty_match=None
    )
    score = compute_quality_score(**inputs)
    assert score.bloom_match == 75.0
    assert score.difficulty_match == 75.0


def test_composite_score_is_weighted_average_within_bounds():
    score = compute_quality_score(**_perfect_inputs())
    assert 0.0 <= score.composite_score <= 100.0
