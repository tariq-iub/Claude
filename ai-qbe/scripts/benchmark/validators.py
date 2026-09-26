"""Automated grading used by the Phase 1 local-LLM benchmark harness.

These are deliberately the *cheap, deterministic* checks (JSON Schema
validity, option-count/word-count rules, exact-match/SymPy correctness,
verifier-verdict accuracy) that don't require a human or an LLM judge.
They are a small, benchmark-scoped preview of the full Phase 7 validation
pipeline (docs/PHASE0-DESIGN.md section 12) — not a replacement for it.

Academic quality (clarity, distractor plausibility, grounding) still needs
the expert-scored rubric in report_template.md; that part is explicitly
NOT automated, per the master prompt's "do not use LLM confidence as the
sole quality metric" rule — a benchmark script judging its own outputs
would have the same problem.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import jsonschema
import sympy
from sympy.parsing.sympy_parser import parse_expr


@dataclass
class GradeResult:
    passed: bool
    score: float  # 0.0-1.0
    details: dict = field(default_factory=dict)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw model output.

    Handles the common failure mode of a model wrapping JSON in markdown
    fences or prefacing it with prose despite instructions not to.
    Returns None (never raises) if nothing parseable is found — a None
    here is itself a benchmark signal (JSON compliance failure), not an
    error to propagate.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def grade_structural(raw_text: str, schema: dict) -> GradeResult:
    """JSON-parseability + JSON-Schema conformance."""
    obj = extract_json(raw_text)
    if obj is None:
        return GradeResult(False, 0.0, {"reason": "not_json_parseable"})
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as exc:
        return GradeResult(False, 0.3, {"reason": "schema_violation", "message": str(exc.message)})
    return GradeResult(True, 1.0, {"parsed": obj})


def grade_option_conciseness(options: list[str], max_words: int = 6) -> GradeResult:
    """Flags options over the 1-6 word target (master prompt section 13)."""
    word_counts = [len(opt.split()) for opt in options]
    over_limit = [wc for wc in word_counts if wc > max_words]
    passed = len(over_limit) == 0
    score = 1.0 - (len(over_limit) / max(len(options), 1))
    return GradeResult(passed, score, {"word_counts": word_counts})


def grade_distractor_hygiene(options: list[str]) -> GradeResult:
    """Cheap structural distractor checks: duplicate/near-duplicate options.

    This is NOT the full Phase 7 distractor validator (which uses
    embedding similarity and off-domain detection) -- it's the
    zero-cost subset worth running inside the benchmark loop itself.
    """
    normalized = [re.sub(r"\s+", " ", o.strip().lower()) for o in options]
    duplicates = len(normalized) - len(set(normalized))
    passed = duplicates == 0
    score = 1.0 if passed else max(0.0, 1.0 - duplicates / len(options))
    return GradeResult(passed, score, {"duplicate_count": duplicates})


def grade_single_correct(mcq: dict) -> GradeResult:
    """Exactly one option must be marked correct via correct_option index."""
    idx = mcq.get("correct_option")
    options = mcq.get("options", [])
    if not isinstance(idx, int) or idx < 0 or idx >= len(options):
        return GradeResult(False, 0.0, {"reason": "correct_option_out_of_range"})
    return GradeResult(True, 1.0, {})


def grade_verification_accuracy(model_output: dict, expected_verdict: str) -> GradeResult:
    """Compares the verifier's PASS/FAIL/UNCERTAIN verdict to ground truth.

    UNCERTAIN is never scored as flatly wrong when the ground truth is
    PASS/FAIL: a verifier that abstains rather than confidently asserting
    a wrong verdict is safer per the master prompt's verification rules,
    so it earns partial credit rather than zero.
    """
    verdict = model_output.get("verdict")
    if verdict == expected_verdict:
        return GradeResult(True, 1.0, {"verdict": verdict})
    if verdict == "UNCERTAIN":
        return GradeResult(False, 0.5, {"verdict": verdict, "reason": "abstained"})
    return GradeResult(False, 0.0, {"verdict": verdict, "reason": "wrong_verdict"})


def grade_math_answer(model_answer_text: str, task: dict) -> GradeResult:
    """Deterministic SymPy check for the math_computation_tasks set.

    The model is expected to have returned a JSON object like
    {"answer": "<sympy-parseable expression or set>"} — see
    run_benchmark.py's prompt wrapper for the exact instruction given
    to the model. This function never trusts the model's own claim of
    correctness (master prompt section 22).
    """
    obj = extract_json(model_answer_text)
    if obj is None or "answer" not in obj:
        return GradeResult(False, 0.0, {"reason": "no_answer_field"})

    x, y = sympy.symbols("x y")
    local_ns = {"x": x, "y": y}

    try:
        op = task["sympy_op"]
        if op in ("diff", "integrate", "simplify", "expand", "factor"):
            given = parse_expr(str(obj["answer"]), local_dict=local_ns)
            expected = parse_expr(task["expected_sympy"], local_dict=local_ns)
            equal = sympy.simplify(given - expected) == 0
        elif op == "definite_integrate":
            given = parse_expr(str(obj["answer"]), local_dict=local_ns)
            expected = parse_expr(task["expected_sympy"], local_dict=local_ns)
            equal = sympy.simplify(given - expected) == 0
        elif op == "limit":
            given = parse_expr(str(obj["answer"]), local_dict=local_ns)
            expected = parse_expr(task["expected_sympy"], local_dict=local_ns)
            equal = sympy.simplify(given - expected) == 0
        elif op == "solve":
            given_set = set(sympy.nsimplify(v) for v in _as_list(obj["answer"]))
            expected_set = set(sympy.nsimplify(v) for v in json.loads(task["expected_sympy"]))
            equal = given_set == expected_set
        elif op == "linsolve":
            equal = _normalize_tuple_set(str(obj["answer"])) == _normalize_tuple_set(
                task["expected_sympy"]
            )
        else:
            return GradeResult(False, 0.0, {"reason": f"unknown_op:{op}"})
    except Exception as exc:  # noqa: BLE001 - any parse/eval failure is a grading fail, not a crash
        return GradeResult(False, 0.0, {"reason": "parse_or_eval_error", "message": str(exc)})

    return GradeResult(bool(equal), 1.0 if equal else 0.0, {"given": str(obj["answer"])})


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    text = str(value).strip("[]() ")
    return [v.strip() for v in text.split(",") if v.strip()]


def _normalize_tuple_set(text: str) -> frozenset:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return frozenset(sympy.nsimplify(n) for n in nums)
