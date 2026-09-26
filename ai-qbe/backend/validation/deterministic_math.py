"""Best-effort deterministic verification of numeric-answer MCQs (master
prompt section 22: "do not rely solely on LLM reasoning for answers that
can be verified computationally").

This is intentionally narrow: it only fires when a clean, unambiguous
arithmetic expression or single-variable equation can be extracted from
the question stem AND every option parses as a bare number. Most
generated questions won't match (conceptual questions, multi-step word
problems phrased in prose, multi-variable systems) and fall through to
the LLM-based verifier pass in answer_verification.py instead --
`applicable=False` signals exactly that, and is never treated as a
verification failure.
"""

from __future__ import annotations

import dataclasses
import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

_ARITHMETIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/^]|times|divided by|plus|minus)\s*(-?\d+(?:\.\d+)?)"
)
_EQUATION_RE = re.compile(r"([0-9a-zA-Z\s\.\+\-\*/\^\(\)]+)=([0-9a-zA-Z\s\.\+\-\*/\^\(\)]+)")
_WORD_OPS = {"times": "*", "divided by": "/", "plus": "+", "minus": "-"}


@dataclasses.dataclass
class DeterministicVerificationResult:
    applicable: bool
    matches: bool | None = None
    computed_value: str | None = None
    reason: str = ""


def _parse_numeric_option(text: str) -> float | None:
    match = re.match(r"^\s*-?\d+(\.\d+)?", text.strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _try_bare_arithmetic(stem: str) -> sympy.core.expr.Expr | None:
    match = _ARITHMETIC_RE.search(stem)
    if not match:
        return None
    left, op, right = match.groups()
    op = _WORD_OPS.get(op, op)
    try:
        return sympy.sympify(f"{left}{op}{right}")
    except (sympy.SympifyError, TypeError):
        return None


def _try_single_variable_equation(stem: str) -> sympy.core.expr.Expr | list | None:
    """Finds a "solve for X" style equation with exactly one free symbol
    and returns its solution set, or None if no clean single-variable
    equation could be extracted.
    """
    for match in _EQUATION_RE.finditer(stem):
        left_text, right_text = match.groups()
        try:
            left = parse_expr(left_text.strip(), transformations=_TRANSFORMATIONS)
            right = parse_expr(right_text.strip(), transformations=_TRANSFORMATIONS)
        except (sympy.SympifyError, TypeError, SyntaxError):
            continue
        free_symbols = (left.free_symbols | right.free_symbols)
        if len(free_symbols) != 1:
            continue
        variable = next(iter(free_symbols))
        try:
            solutions = sympy.solve(sympy.Eq(left, right), variable)
        except (NotImplementedError, TypeError):
            continue
        if solutions:
            return solutions
    return None


def verify_numeric_mcq(question_stem: str, options: list[str], correct_option: int) -> DeterministicVerificationResult:
    numeric_options = [_parse_numeric_option(o) for o in options]
    if any(v is None for v in numeric_options):
        return DeterministicVerificationResult(applicable=False, reason="not_all_options_numeric")

    claimed_value = numeric_options[correct_option]

    # Equation-solving is tried first: a stem like "x^2 - 5x + 6 = 0"
    # would otherwise be misdetected by the bare-arithmetic pattern (which
    # matches any "NUM OP NUM" substring) before reaching the solver.
    solutions = _try_single_variable_equation(question_stem)
    if solutions:
        try:
            numeric_solutions = [float(s) for s in solutions if s.is_real]
        except (TypeError, AttributeError, ValueError):
            return DeterministicVerificationResult(applicable=False, reason="equation_solution_not_numeric")
        if numeric_solutions:
            claimed_matches = any(abs(claimed_value - s) < 1e-6 for s in numeric_solutions)
            return DeterministicVerificationResult(
                applicable=True,
                matches=claimed_matches,
                computed_value=", ".join(str(s) for s in numeric_solutions),
                reason="single_variable_equation_solved",
            )

    computed = _try_bare_arithmetic(question_stem)
    if computed is not None:
        try:
            computed_float = float(computed)
        except (TypeError, ValueError):
            return DeterministicVerificationResult(applicable=False, reason="arithmetic_result_not_numeric")
        matches = any(abs(computed_float - v) < 1e-6 for v in numeric_options if v is not None)
        claimed_matches = abs(computed_float - claimed_value) < 1e-6
        return DeterministicVerificationResult(
            applicable=True,
            matches=claimed_matches,
            computed_value=str(computed_float),
            reason="bare_arithmetic_extracted_from_stem" if matches else "computed_value_matches_no_option",
        )

    return DeterministicVerificationResult(applicable=False, reason="no_extractable_computation")
