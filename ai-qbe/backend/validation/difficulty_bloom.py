"""Difficulty and Bloom-level cross-checking (docs/PHASE0-DESIGN.md
section 12: "cross-checked against the LLM's self-reported label;
mismatch beyond a threshold flags for review rather than silently
overriding").

Deliberately a checklist-style heuristic, not a learned classifier
(master prompt section 11: "do not estimate difficulty solely from
sentence length"). A mismatch never overrides the generator's declared
label or blocks the candidate -- it's recorded so a human reviewer (who
already has "change_difficulty"/"change_bloom" actions available, per
Phase 2) sees the disagreement instead of it being silently discarded.
"""

from __future__ import annotations

import dataclasses
import re

from backend.domain.enums import BloomLevel, DifficultyLevel

_BLOOM_KEYWORDS = {
    BloomLevel.REMEMBER: ("define", "what is", "identify", "name", "list", "state", "which term", "recall"),
    BloomLevel.UNDERSTAND: ("explain", "describe", "why does", "summarize", "interpret", "which best describes"),
    BloomLevel.APPLY: ("calculate", "compute", "solve", "apply", "determine", "find the value", "what is the result"),
    BloomLevel.ANALYZE: ("compare", "analyze", "evaluate", "predict", "differentiate", "which factor", "what would happen if"),
}

_CALCULATION_INDICATORS = re.compile(r"[=+\-*/^]|\d+\s*(m/s|kg|N|J|W|Pa|mol|°)")


@dataclasses.dataclass
class DifficultyBloomCheckResult:
    estimated_bloom: BloomLevel | None
    estimated_difficulty: DifficultyLevel | None
    bloom_match: bool | None  # None if no confident estimate could be made
    difficulty_match: bool | None
    notes: list[str] = dataclasses.field(default_factory=list)


def estimate_bloom(question_stem: str) -> BloomLevel | None:
    lowered = question_stem.lower()
    for level, keywords in _BLOOM_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return level
    return None


def estimate_difficulty(question_stem: str, options: list[str]) -> DifficultyLevel:
    """Checklist-scored estimate combining: presence of a calculation,
    word-problem length (multi-sentence stems tend to require more
    reasoning steps), and distractor similarity (options that are close
    in form to each other suggest a harder discrimination) -- not raw
    sentence length alone.
    """
    score = 0
    has_calculation = bool(_CALCULATION_INDICATORS.search(question_stem))
    if has_calculation:
        score += 1

    sentence_count = len(re.findall(r"[.!?]+", question_stem))
    if sentence_count >= 2:
        score += 1  # multi-sentence stem suggests a scenario/multi-step problem

    numeric_options = sum(1 for o in options if re.match(r"^-?\d", o.strip()))
    if numeric_options == len(options) and len(options) > 0:
        score += 1  # a fully-numeric option set usually means a computed answer

    if score >= 2:
        return DifficultyLevel.HARD
    if score == 1:
        return DifficultyLevel.MEDIUM
    return DifficultyLevel.EASY


def check_difficulty_bloom(
    question_stem: str,
    options: list[str],
    declared_bloom: BloomLevel,
    declared_difficulty: DifficultyLevel,
) -> DifficultyBloomCheckResult:
    estimated_bloom = estimate_bloom(question_stem)
    estimated_difficulty = estimate_difficulty(question_stem, options)

    notes = []
    bloom_match = None
    if estimated_bloom is not None:
        bloom_match = estimated_bloom == declared_bloom
        if not bloom_match:
            notes.append(f"declared bloom '{declared_bloom.value}' but stem phrasing suggests '{estimated_bloom.value}'")

    difficulty_match = estimated_difficulty == declared_difficulty
    if not difficulty_match:
        notes.append(
            f"declared difficulty '{declared_difficulty.value}' but heuristic estimate is '{estimated_difficulty.value}'"
        )

    return DifficultyBloomCheckResult(
        estimated_bloom=estimated_bloom,
        estimated_difficulty=estimated_difficulty,
        bloom_match=bloom_match,
        difficulty_match=difficulty_match,
        notes=notes,
    )
