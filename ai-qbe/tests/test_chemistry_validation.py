from backend.validation.chemistry import (
    looks_like_reaction_equation,
    strip_ce_wrapper,
    validate_equation_balance,
    validate_formula,
)


def test_strip_ce_wrapper_unwraps():
    assert strip_ce_wrapper("\\ce{H2O}") == "H2O"


def test_strip_ce_wrapper_passthrough_when_not_wrapped():
    assert strip_ce_wrapper("H2O") == "H2O"


def test_valid_simple_formula():
    result = validate_formula("H2O")
    assert result.passed
    assert result.element_counts == {"H": 2, "O": 1}


def test_valid_formula_with_parentheses():
    result = validate_formula("Ca(OH)2")
    assert result.passed
    assert result.element_counts == {"Ca": 1, "O": 2, "H": 2}


def test_valid_ion_with_charge():
    result = validate_formula("SO4^2-")
    assert result.passed
    assert result.element_counts == {"S": 1, "O": 4}


def test_valid_ce_wrapped_formula():
    result = validate_formula("\\ce{NH4+}")
    assert result.passed
    assert result.element_counts == {"N": 1, "H": 4}


def test_unknown_element_symbol_fails():
    result = validate_formula("Xx2O")
    assert not result.passed
    assert any("unknown_element_symbol" in r for r in result.reasons)


def test_unbalanced_parentheses_fails():
    result = validate_formula("Ca(OH2")
    assert not result.passed
    assert "unbalanced_parentheses" in result.reasons


def test_garbage_input_fails():
    result = validate_formula("not a formula!!")
    assert not result.passed


def test_bare_digit_before_charge_sign_is_treated_as_subscript_not_magnitude():
    """Documented scope decision: "Fe3+" without a caret parses as Fe with
    subscript 3 and charge +1, not "charge magnitude 3" -- ambiguous
    without a caret, and this is the convention that correctly handles
    the far more common "NH4+" (ammonium: subscript 4 on H, charge +1)
    case the same way. A magnitude>1 charge on a monatomic ion is
    expected to use the explicit caret form, \\ce{Fe^3+}."""
    result = validate_formula("Fe3+")
    assert result.passed
    assert result.element_counts == {"Fe": 3}

    caret_result = validate_formula("\\ce{Fe^3+}")
    assert caret_result.passed
    assert caret_result.element_counts == {"Fe": 1}


def test_lowercase_leading_letter_fails():
    # Element symbols always start uppercase; "h2o" is not valid notation.
    result = validate_formula("h2o")
    assert not result.passed


def test_balanced_combustion_equation():
    result = validate_equation_balance("2H2 + O2 -> 2H2O")
    assert result.balanced
    assert result.left_totals == result.right_totals == {"H": 4, "O": 2}


def test_unbalanced_combustion_equation():
    result = validate_equation_balance("H2 + O2 -> H2O")
    assert not result.balanced
    assert any("unbalanced_elements" in r for r in result.reasons)


def test_balanced_equation_with_ce_wrapper():
    result = validate_equation_balance("\\ce{2H2 + O2 -> 2H2O}")
    assert result.balanced


def test_balanced_equation_with_equals_sign_form():
    result = validate_equation_balance("2Na + Cl2 = 2NaCl")
    assert result.balanced


def test_equation_missing_arrow_fails():
    result = validate_equation_balance("H2 + O2")
    assert not result.balanced
    assert "no_reaction_arrow_found" in result.reasons


def test_equation_with_unknown_element_reports_error():
    result = validate_equation_balance("2Xx + O2 -> 2XxO")
    assert not result.balanced


def test_looks_like_reaction_equation():
    assert looks_like_reaction_equation("2H2 + O2 -> 2H2O")
    assert looks_like_reaction_equation("A <-> B")
    assert not looks_like_reaction_equation("H2O")


def test_more_complex_balanced_equation():
    # Combustion of methane: CH4 + 2O2 -> CO2 + 2H2O
    result = validate_equation_balance("CH4 + 2O2 -> CO2 + 2H2O")
    assert result.balanced
    assert result.left_totals == {"C": 1, "H": 4, "O": 4}
    assert result.right_totals == {"C": 1, "O": 4, "H": 4}
