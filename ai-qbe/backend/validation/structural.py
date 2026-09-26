"""Structural validation for freshly-generated MCQ candidates.

This is the first stage of the Phase 7 validation pipeline
(docs/PHASE0-DESIGN.md section 12), pulled forward into Phase 2 because a
generation job cannot responsibly write a candidate to the database
without at least confirming its shape is sane -- schema conformance,
option count, exactly one correct answer, option-length limits, no
duplicate options. Fact-checking (independent answer verification),
source-grounding, semantic deduplication, and full quality scoring are
still Phase 7 work and are explicitly NOT done here.

The generation executor treats a StructuralValidationResult with
`passed=False` as a reason to mark the candidate INVALID rather than ever
writing an unusable row to mcq_candidates as if it were reviewable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jsonschema


@dataclass
class StructuralValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def validate_mcq_structure(
    candidate: dict, schema: dict, *, max_option_words: int = 6
) -> StructuralValidationResult:
    reasons: list[str] = []

    try:
        jsonschema.validate(candidate, schema)
    except jsonschema.ValidationError as exc:
        return StructuralValidationResult(False, [f"schema_violation: {exc.message}"])

    options = candidate["options"]
    correct_idx = candidate["correct_option"]

    if not (0 <= correct_idx < len(options)):
        reasons.append("correct_option_out_of_range")

    normalized = [re.sub(r"\s+", " ", opt.strip().lower()) for opt in options]
    if len(normalized) != len(set(normalized)):
        reasons.append("duplicate_options")

    word_counts = [len(opt.split()) for opt in options]
    over_limit = [i for i, wc in enumerate(word_counts) if wc > max_option_words]
    if over_limit:
        reasons.append(f"options_exceed_{max_option_words}_words: indices {over_limit}")

    # The JSON Schema's own minLength (8 chars) only rules out truly empty
    # stems; this stricter check catches degenerate-but-schema-valid stems
    # like "Why now?" that are still too short to be a usable question.
    if len(candidate["question"].strip()) < 15:
        reasons.append("question_stem_too_short")

    return StructuralValidationResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        details={"word_counts": word_counts},
    )
