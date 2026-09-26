"""Chemical formula and reaction-equation validation (docs/PHASE0-DESIGN.md
section 13/18-19, master prompt sections 18-19).

Independent of MathJax rendering: `\\ce{H2SO4}` can render without a
MathJax error while still being chemical nonsense (an unbalanced
equation, an unknown element symbol, a charge notation that doesn't
parse). This module checks the chemistry *content*, not the LaTeX syntax
-- both checks run, and either one failing is enough to flag the
notation as invalid (see backend/validation/notation.py).

Parses plain chemical notation ("H2O", "SO4^2-") and mhchem's `\\ce{}`
argument content, since master prompt section 18 leaves both as valid
storage forms depending on whether mhchem is enabled for a deployment.
"""

from __future__ import annotations

import dataclasses
import re

# IUPAC element symbols, periods 1-7 (all 118). Kept as a flat set rather
# than a full periodic-table data structure since only membership is
# needed here; atomic number/mass would belong in a different module if
# ever needed (e.g. a future molar-mass calculator).
ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
}

# Matches one element+optional-subscript token, e.g. "H2", "Na", "Cl2".
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
# A trailing charge marker. Two forms are recognized:
#   - explicit superscript, with an optional magnitude digit: "^2-", "^+",
#     "^3+", "^{2-}" -- the digit here is unambiguously the charge
#     magnitude;
#   - a bare trailing run of sign characters with NO caret: "+", "-",
#     "++", "---" (magnitude = number of repeated signs).
# A digit immediately before a bare (non-caret) sign is always treated as
# the PRECEDING ELEMENT's subscript, never as charge magnitude -- e.g.
# "NH4+" parses as subscript 4 on H, charge +1 (ammonium), matching
# standard tokenization. This means a magnitude-greater-than-1 charge on
# a monatomic ion written without a caret ("Fe3+" intending charge +3)
# instead parses as Fe with subscript 3 and charge +1 -- a documented
# scope limitation, not a silent guess: standard mhchem/IUPAC practice
# already expects the explicit caret form (\ce{Fe^3+}) for exactly this
# reason.
_CHARGE_RE = re.compile(r"\^\{?(\d*)([+\-]+)\}?$|([+\-]+)$")


@dataclasses.dataclass
class FormulaValidationResult:
    passed: bool
    reasons: list[str] = dataclasses.field(default_factory=list)
    element_counts: dict[str, int] = dataclasses.field(default_factory=dict)


def strip_ce_wrapper(text: str) -> str:
    """Unwraps a `\\ce{...}` argument if present, else returns `text`
    unchanged (plain "H2O"-style notation is also valid, per master
    prompt section 18's "portable textual form")."""
    match = re.match(r"^\\ce\{(.*)\}$", text.strip(), re.DOTALL)
    return match.group(1) if match else text.strip()


def parse_formula(formula: str) -> tuple[dict[str, int], list[str]]:
    """Parses a single-species formula (no reaction arrows) like "H2SO4"
    or "SO4^2-" into {element: count}. Returns (counts, errors); a
    non-empty errors list means the formula could not be fully parsed
    (e.g. contains characters that aren't a recognized element symbol,
    unbalanced parentheses).
    """
    errors: list[str] = []
    body = _CHARGE_RE.sub("", formula.strip())

    if body.count("(") != body.count(")"):
        errors.append("unbalanced_parentheses")
        return {}, errors

    # Expand parenthesized groups: (OH)2 -> OH OH, applied innermost-first.
    paren_re = re.compile(r"\(([^()]*)\)(\d*)")
    while "(" in body:
        new_body, count = paren_re.subn(
            lambda m: (m.group(1) * int(m.group(2) or 1)), body
        )
        if count == 0:
            errors.append("unbalanced_parentheses")
            return {}, errors
        body = new_body

    counts: dict[str, int] = {}
    pos = 0
    while pos < len(body):
        match = _FORMULA_TOKEN_RE.match(body, pos)
        if not match or match.start() != pos:
            errors.append(f"unrecognized_token_at_position_{pos}: '{body[pos:pos+4]}'")
            break
        symbol, count_str = match.groups()
        if symbol not in ELEMENT_SYMBOLS:
            errors.append(f"unknown_element_symbol: '{symbol}'")
            break
        count = int(count_str) if count_str else 1
        counts[symbol] = counts.get(symbol, 0) + count
        pos = match.end()

    return counts, errors


def validate_formula(formula: str) -> FormulaValidationResult:
    body = strip_ce_wrapper(formula)
    counts, errors = parse_formula(body)

    charge_match = _CHARGE_RE.search(body)
    if charge_match:
        magnitude = charge_match.group(1)  # only set for the caret form; bare-sign form has no magnitude group
        if magnitude and int(magnitude) == 0:
            errors.append("zero_magnitude_charge")

    return FormulaValidationResult(passed=len(errors) == 0, reasons=errors, element_counts=counts)


@dataclasses.dataclass
class EquationBalanceResult:
    balanced: bool
    reasons: list[str] = dataclasses.field(default_factory=list)
    left_totals: dict[str, int] = dataclasses.field(default_factory=dict)
    right_totals: dict[str, int] = dataclasses.field(default_factory=dict)


_ARROW_RE = re.compile(r"->|<->|\\rightarrow|\\rightleftharpoons|=")


def _parse_side(side: str) -> tuple[dict[str, int], list[str]]:
    totals: dict[str, int] = {}
    errors: list[str] = []
    for term in side.split("+"):
        term = term.strip()
        if not term:
            continue
        coeff_match = re.match(r"^(\d+)\s*(.*)$", term)
        coeff = int(coeff_match.group(1)) if coeff_match else 1
        formula = coeff_match.group(2) if coeff_match else term
        counts, term_errors = parse_formula(strip_ce_wrapper(formula))
        if term_errors:
            errors.extend(f"{formula}: {e}" for e in term_errors)
            continue
        for element, count in counts.items():
            totals[element] = totals.get(element, 0) + coeff * count
    return totals, errors


def validate_equation_balance(equation: str) -> EquationBalanceResult:
    """Checks that a chemical equation ("2H2 + O2 -> 2H2O" or the same
    inside `\\ce{}`) has equal atom counts of every element on both
    sides -- independent of whether the LaTeX/mhchem syntax itself is
    valid (that's mathjax_bridge.py's job).
    """
    body = strip_ce_wrapper(equation)
    match = _ARROW_RE.search(body)
    if not match:
        return EquationBalanceResult(False, ["no_reaction_arrow_found"])

    left, right = body[: match.start()], body[match.end() :]
    left_totals, left_errors = _parse_side(left)
    right_totals, right_errors = _parse_side(right)

    reasons = list(left_errors) + list(right_errors)
    if not reasons and left_totals != right_totals:
        all_elements = set(left_totals) | set(right_totals)
        mismatched = [
            e for e in all_elements if left_totals.get(e, 0) != right_totals.get(e, 0)
        ]
        reasons.append(f"unbalanced_elements: {sorted(mismatched)}")

    return EquationBalanceResult(
        balanced=len(reasons) == 0, reasons=reasons, left_totals=left_totals, right_totals=right_totals
    )


def looks_like_reaction_equation(text: str) -> bool:
    return bool(_ARROW_RE.search(text))
