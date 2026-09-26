"""Tests backend/validation/notation.py, including the Phase 6 acceptance
criterion (docs/PHASE0-DESIGN.md section 20): "notation validator rejects
a seeded set of malformed LaTeX/chemical-formula test cases with zero
false negatives on that set" -- and, symmetrically, zero false positives
on a seeded set of valid academic notation covering math, physics, and
chemistry per master prompt sections 16-18.
"""

import pytest

from backend.validation.mathjax_bridge import is_available
from backend.validation.notation import validate_notation

pytestmark = pytest.mark.skipif(
    not is_available(), reason="Node.js + tools/mathjax_validator's mathjax-full dependency not installed"
)


def _mcq(question: str, options: list[str], explanation: str = "") -> dict:
    return {"question": question, "options": options, "explanation": explanation}


# --- Valid academic notation across math, physics, chemistry -----------

VALID_CASES = [
    _mcq("What is $x^2+y^2$ when $x=3, y=4$?", ["25", "12", "7", "49"]),
    _mcq("Evaluate $\\int_0^1 x^2\\,dx$.", ["$\\frac{1}{3}$", "$\\frac{1}{2}$", "$1$", "$0$"]),
    _mcq("What does $F=ma$ describe?", ["Newton's second law", "Ohm's law", "Hooke's law", "Boyle's law"]),
    _mcq("A vector sum: $\\vec{A}+\\vec{B}$ has what property?", ["Commutative", "Associative only", "Undefined", "Zero"]),
    _mcq(
        "Balance: $\\ce{2H2 + O2 -> 2H2O}$. What does this represent?",
        ["Combustion of hydrogen", "Electrolysis", "Neutralization", "Sublimation"],
    ),
    _mcq("What ion is $\\ce{SO4^2-}$?", ["Sulfate", "Sulfite", "Sulfide", "Sulfur"]),
    _mcq("Matrix product uses $\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}$ notation for what?", ["Linear transformations", "Scalars", "Sets", "Graphs"]),
    _mcq("Greek letters like $\\alpha, \\beta, \\gamma$ commonly denote what in physics?", ["Angles or coefficients", "Units", "Vectors always", "Nothing"]),
    _mcq("What is $6.022 \\times 10^{23}$ known as?", ["Avogadro's number", "Planck's constant", "Boltzmann's constant", "Faraday's constant"]),
    _mcq("Simplify $\\sum_{i=1}^n i$.", ["$\\frac{n(n+1)}{2}$", "$n^2$", "$n$", "$2n$"]),
]

# --- Malformed / invalid notation ---------------------------------------

MALFORMED_CASES = [
    ("unbalanced_brace_in_question", _mcq("Evaluate $\\frac{a}{b$.", ["1", "2", "3", "4"])),
    ("unbalanced_brace_in_option", _mcq("Simplify the expression.", ["$\\frac{1}{2$", "1", "2", "3"])),
    ("undefined_command", _mcq("What is $\\notarealcommand{x}$?", ["A", "B", "C", "D"])),
    ("bad_matrix_environment", _mcq("Consider $\\begin{bmatrix} a & b \\end{matrx}$.", ["A", "B", "C", "D"])),
    (
        "unbalanced_chemical_equation",
        _mcq("Which equation is shown: $\\ce{H2 + O2 -> H2O}$?", ["Combustion", "Neutralization", "Decomposition", "None"]),
    ),
    ("unknown_element_symbol", _mcq("Identify $\\ce{Xx2O}$.", ["A", "B", "C", "D"])),
    (
        "unbalanced_parentheses_formula",
        _mcq("What is $\\ce{Ca(OH2}$?", ["A", "B", "C", "D"]),
    ),
    ("malformed_in_explanation", _mcq("Which law is this?", ["Second law", "First law", "Third law", "Zeroth law"], explanation="Because $\\frac{F}{m = a$.")),
]


@pytest.mark.parametrize("case", VALID_CASES, ids=[c["question"][:30] for c in VALID_CASES])
def test_valid_notation_cases_all_pass(case):
    result = validate_notation(case)
    assert result.passed, f"false positive: {result.reasons}"


@pytest.mark.parametrize("name,case", MALFORMED_CASES, ids=[c[0] for c in MALFORMED_CASES])
def test_malformed_notation_cases_all_rejected(name, case):
    result = validate_notation(case)
    assert not result.passed, f"false negative for {name}: expected rejection but got passed=True"
    assert result.reasons  # a real, inspectable reason was recorded, not a bare False


def test_no_math_content_trivially_passes():
    result = validate_notation(_mcq("What is the capital of France?", ["Paris", "London", "Berlin", "Rome"]))
    assert result.passed
    assert result.status_label == "pass"
    assert not result.had_math_content


def test_status_label_pass_for_clean_notation():
    result = validate_notation(VALID_CASES[0])
    assert result.status_label == "pass"


def test_status_label_fail_for_malformed_notation():
    result = validate_notation(MALFORMED_CASES[0][1])
    assert result.status_label == "fail"


def test_unit_warnings_flag_missing_units_without_failing():
    case = _mcq("What is the acceleration?", ["9.8", "9.8 m/s^2", "10", "gravity"])
    result = validate_notation(case)
    assert result.passed  # unit warnings never fail the candidate
    assert any("option_0" in w for w in result.unit_warnings)
    assert not any("option_1" in w for w in result.unit_warnings)
