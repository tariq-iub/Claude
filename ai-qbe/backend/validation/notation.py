"""Scientific notation validation for a generated MCQ candidate
(docs/PHASE0-DESIGN.md section 19 / master prompt section 19: "No
malformed notation should automatically enter the approved bank").

Combines three independent checks, none of which substitutes for the
others:
  1. MathJax rendering (backend/validation/mathjax_bridge.py) -- catches
     malformed LaTeX syntax: unbalanced braces, undefined commands,
     unsupported environments.
  2. Chemistry content validation (backend/validation/chemistry.py) --
     catches chemically-wrong content that is still syntactically valid
     LaTeX/mhchem: an unknown element symbol, an unbalanced reaction.
  3. Physics unit presence (backend/validation/units.py) -- a soft signal
     only (warnings, not failures): flags a numeric option that looks
     like it's missing a unit, without asserting the missing unit means
     the notation itself is malformed.

A candidate with NO math notation anywhere in question/options/
explanation trivially passes (there is nothing to validate) -- this
module doesn't force every question to contain LaTeX.
"""

from __future__ import annotations

import dataclasses

from backend.validation import chemistry
from backend.validation.latex_extraction import extract_math_segments
from backend.validation.mathjax_bridge import MathJaxUnavailableError, render_check_batch
from backend.validation.units import check_numeric_option_has_unit


@dataclasses.dataclass
class NotationValidationResult:
    passed: bool
    mathjax_checked: bool
    had_math_content: bool
    reasons: list[str] = dataclasses.field(default_factory=list)
    unit_warnings: list[str] = dataclasses.field(default_factory=list)
    details: dict = dataclasses.field(default_factory=dict)

    @property
    def status_label(self) -> str:
        """PASS/FAIL/UNCERTAIN, mirroring the answer-verification schema's
        three-way outcome (docs/PHASE0-DESIGN.md section 21): a candidate
        containing math that couldn't be checked (MathJax unavailable) is
        UNCERTAIN, never silently recorded the same as a real PASS. Any
        reason present (chemistry content is always checkable regardless
        of MathJax availability) makes it a definite FAIL.
        """
        if self.reasons:
            return "fail"
        if self.had_math_content and not self.mathjax_checked:
            return "uncertain"
        return "pass"


def _collect_text_fields(mcq: dict) -> dict[str, str]:
    fields = {"question": mcq.get("question", ""), "explanation": mcq.get("explanation", "") or ""}
    for i, option in enumerate(mcq.get("options", [])):
        fields[f"option_{i}"] = option
    return fields


def validate_notation(mcq: dict) -> NotationValidationResult:
    fields = _collect_text_fields(mcq)

    all_segments: list[tuple[str, str]] = []  # (field_name, latex)
    for field_name, text in fields.items():
        for segment in extract_math_segments(text):
            all_segments.append((field_name, segment.latex))

    reasons: list[str] = []
    mathjax_checked = False
    details: dict = {}

    if all_segments:
        try:
            render_results = render_check_batch([latex for _, latex in all_segments])
            mathjax_checked = True
            for (field_name, latex), result in zip(all_segments, render_results):
                if not result.ok:
                    reasons.append(f"{field_name}: malformed LaTeX '{latex}' -- {result.error}")
        except MathJaxUnavailableError as exc:
            # Recorded explicitly via mathjax_checked=False -- the syntax
            # check is skipped, not silently counted as passed. Chemistry
            # content checks below are pure-Python and still run
            # regardless, since they don't depend on the Node subprocess.
            details["mathjax_skipped_reason"] = str(exc)

    # Chemistry content checks run on \ce{...}-wrapped segments regardless
    # of whether MathJax itself was reachable -- these are pure-Python
    # parses, independent of the Node subprocess.
    for field_name, latex in all_segments:
        stripped = latex.strip()
        if not stripped.startswith("\\ce{"):
            continue
        if chemistry.looks_like_reaction_equation(chemistry.strip_ce_wrapper(stripped)):
            balance = chemistry.validate_equation_balance(stripped)
            if not balance.balanced:
                reasons.append(f"{field_name}: unbalanced chemical equation '{stripped}' -- {balance.reasons}")
        else:
            formula = chemistry.validate_formula(stripped)
            if not formula.passed:
                reasons.append(f"{field_name}: invalid chemical formula '{stripped}' -- {formula.reasons}")

    unit_warnings = []
    for i, option in enumerate(mcq.get("options", [])):
        has_unit = check_numeric_option_has_unit(option)
        if has_unit is False:
            unit_warnings.append(f"option_{i}: numeric value '{option}' has no recognized unit")

    return NotationValidationResult(
        passed=len(reasons) == 0,
        mathjax_checked=mathjax_checked,
        had_math_content=len(all_segments) > 0,
        reasons=reasons,
        unit_warnings=unit_warnings,
        details=details,
    )
