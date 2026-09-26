"""Independent answer verification (docs/PHASE0-DESIGN.md section 12/21,
master prompt sections 21-22): never trust the generator's own claimed
correct answer.

Two strategies, tried in order:
  1. Deterministic (backend/validation/deterministic_math.py) -- SymPy-
     based, applies only to a narrow but common case (a clean arithmetic
     expression or single-variable equation extractable from the stem,
     with every option a bare number). When it applies, its answer is
     authoritative and no LLM call is needed.
  2. LLM verifier pass -- a second, independent prompt (never the
     generation prompt/context reused verbatim) asking the model to
     derive the correct answer itself and compare, using the same
     PASS/FAIL/UNCERTAIN verification_schema.json from Phase 1's
     benchmark harness. An `UNCERTAIN` verdict is a legitimate outcome,
     not a failure of the verifier -- it means "send to human review",
     never "assume correct."

Whichever strategy ran, this always returns one of PASS/FAIL/UNCERTAIN;
callers must never downgrade UNCERTAIN to an implicit PASS.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from backend.validation.deterministic_math import verify_numeric_mcq
from llm.providers.base import ILLMProvider
from rag.web.injection_defense import wrap_context_as_data

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "llm" / "schemas" / "verification_schema.json"
VERIFICATION_SCHEMA = json.loads(_SCHEMA_PATH.read_text())


@dataclasses.dataclass
class AnswerVerificationResult:
    verdict: str  # "PASS" | "FAIL" | "UNCERTAIN"
    method: str  # "deterministic" | "llm"
    confidence: float | None = None
    reason: str = ""
    derived_correct_option: int | None = None


def verify_answer(
    provider: ILLMProvider,
    *,
    question_stem: str,
    options: list[str],
    correct_option: int,
    context: str = "",
) -> AnswerVerificationResult:
    deterministic = verify_numeric_mcq(question_stem, options, correct_option)
    if deterministic.applicable:
        return AnswerVerificationResult(
            verdict="PASS" if deterministic.matches else "FAIL",
            method="deterministic",
            confidence=1.0,
            reason=f"{deterministic.reason} (computed: {deterministic.computed_value})",
        )

    options_text = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
    context_block = wrap_context_as_data(context) if context else "(no supporting context supplied)"
    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question_stem}\n"
        f"Options:\n{options_text}\n\n"
        f"A generator claims option index {correct_option} is correct. Independently "
        f"derive the correct answer from first principles and the supplied context "
        f"only. Decide PASS if the claim is correct, FAIL if it is incorrect, or "
        f"UNCERTAIN if the context does not give you enough to decide confidently. "
        f"Never assume the claim is correct by default."
    )

    result = provider.generate_structured(prompt, json_schema=VERIFICATION_SCHEMA, temperature=0.1, max_tokens=400)
    parsed = _try_parse_json(result.text)
    if parsed is None or parsed.get("verdict") not in ("PASS", "FAIL", "UNCERTAIN"):
        return AnswerVerificationResult(
            verdict="UNCERTAIN", method="llm", reason="verifier_returned_unparseable_or_invalid_response"
        )

    return AnswerVerificationResult(
        verdict=parsed["verdict"],
        method="llm",
        confidence=parsed.get("confidence"),
        reason=parsed.get("reason", ""),
        derived_correct_option=parsed.get("derived_correct_option"),
    )


def _try_parse_json(text: str) -> dict | None:
    import re

    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
