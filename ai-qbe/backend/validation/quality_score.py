"""Composite quality scoring (docs/PHASE0-DESIGN.md section 12, master
prompt section 25): "Quality must derive from measurable validation
results," never from the LLM's own self-reported confidence.

Every sub-score here is computed from an upstream validator's actual
output (structural, notation, answer verification, distractor, dedup,
difficulty/Bloom match) -- `raw_item["confidence"]` (the generator's own
self-assessment) never appears in this module, deliberately.
"""

from __future__ import annotations

import dataclasses

from backend.validation.answer_verification import AnswerVerificationResult
from backend.validation.dedup import DedupResult
from backend.validation.difficulty_bloom import DifficultyBloomCheckResult
from backend.validation.distractor import DistractorValidationResult
from backend.validation.notation import NotationValidationResult
from backend.validation.structural import StructuralValidationResult

# Weights sum to 1.0. Factual correctness and single-correctness (a
# multiple-correct-answer question is a fundamentally broken item) are
# weighted highest since they're the dimensions most likely to make a
# question simply wrong, not just imperfect.
_WEIGHTS = {
    "factual_correctness": 0.25,
    "source_grounding": 0.10,
    "clarity": 0.10,
    "distractor_quality": 0.15,
    "single_correctness": 0.15,
    "difficulty_match": 0.05,
    "bloom_match": 0.05,
    "option_conciseness": 0.05,
    "notation_validity": 0.05,
    "duplicate_risk": 0.05,
}


@dataclasses.dataclass
class QualityScore:
    factual_correctness: float
    source_grounding: float
    clarity: float
    distractor_quality: float
    single_correctness: float
    difficulty_match: float
    bloom_match: float
    option_conciseness: float
    notation_validity: float
    duplicate_risk: float
    composite_score: float


def _answer_verification_score(result: AnswerVerificationResult) -> float:
    if result.verdict == "PASS":
        return 100.0
    if result.verdict == "UNCERTAIN":
        return 50.0
    return 0.0  # FAIL should already have blocked the candidate before scoring is reached


def _source_grounding_score(has_sources: bool, used_manual_context: bool, had_any_context: bool) -> float:
    if has_sources:
        return 100.0  # grounded in specific, citable retrieved evidence
    if used_manual_context and had_any_context:
        return 60.0  # grounded in supplied context, but no per-chunk citation trail
    return 20.0  # ungrounded -- generated with no evidence at all


def _distractor_score(result: DistractorValidationResult) -> float:
    score = result.score
    if not result.embedding_checked:
        # An unverified distractor check should not read as confidently
        # perfect -- cap it so "we didn't check" is visibly different
        # from "we checked and it's great" in the composite score.
        score = min(score, 75.0)
    return score


def _duplicate_risk_score(result: DedupResult) -> float:
    if result.is_duplicate:
        return 0.0  # should already have blocked the candidate before scoring
    if result.score is not None and result.score >= 0.70:
        return 100.0 - (result.score - 0.70) * 200  # borderline-but-cleared gets a lower score, not a perfect one
    return 100.0


def _bool_or_none_score(value: bool | None) -> float:
    if value is None:
        return 75.0  # no confident estimate either way -- neutral, not penalized
    return 100.0 if value else 50.0


def compute_quality_score(
    *,
    structural: StructuralValidationResult,
    notation: NotationValidationResult,
    answer_verification: AnswerVerificationResult,
    distractor: DistractorValidationResult,
    dedup: DedupResult,
    difficulty_bloom: DifficultyBloomCheckResult,
    has_sources: bool,
    used_manual_context: bool,
    had_any_context: bool,
) -> QualityScore:
    factual_correctness = _answer_verification_score(answer_verification)
    source_grounding = _source_grounding_score(has_sources, used_manual_context, had_any_context)
    clarity = 100.0 if structural.passed else 0.0
    distractor_quality = _distractor_score(distractor)
    single_correctness = 100.0 if structural.passed and distractor.passed else 0.0
    difficulty_match = _bool_or_none_score(difficulty_bloom.difficulty_match)
    bloom_match = _bool_or_none_score(difficulty_bloom.bloom_match)
    option_conciseness = 100.0 if structural.passed else max(0.0, 100.0 - 20 * len(structural.reasons))
    notation_validity = 100.0 if notation.passed else 0.0
    duplicate_risk = _duplicate_risk_score(dedup)

    values = {
        "factual_correctness": factual_correctness,
        "source_grounding": source_grounding,
        "clarity": clarity,
        "distractor_quality": distractor_quality,
        "single_correctness": single_correctness,
        "difficulty_match": difficulty_match,
        "bloom_match": bloom_match,
        "option_conciseness": option_conciseness,
        "notation_validity": notation_validity,
        "duplicate_risk": duplicate_risk,
    }
    composite = sum(values[k] * _WEIGHTS[k] for k in _WEIGHTS)

    return QualityScore(composite_score=composite, **values)
