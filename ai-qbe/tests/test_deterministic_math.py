from backend.validation.deterministic_math import verify_numeric_mcq


def test_bare_arithmetic_correct_claim_passes():
    result = verify_numeric_mcq("What is 12 + 8?", ["20", "18", "22", "16"], 0)
    assert result.applicable
    assert result.matches is True


def test_bare_arithmetic_wrong_claim_fails():
    result = verify_numeric_mcq("What is 12 + 8?", ["20", "18", "22", "16"], 1)
    assert result.applicable
    assert result.matches is False


def test_single_variable_equation_correct_claim_passes():
    result = verify_numeric_mcq("Solve for x: 2x + 4 = 10", ["3", "2", "4", "5"], 0)
    assert result.applicable
    assert result.matches is True


def test_single_variable_equation_wrong_claim_fails():
    result = verify_numeric_mcq("Solve for x: 2x + 4 = 10", ["3", "2", "4", "5"], 2)
    assert result.applicable
    assert result.matches is False


def test_multiplication_arithmetic():
    result = verify_numeric_mcq("What is 6 times 7?", ["42", "48", "36", "40"], 0)
    assert result.applicable
    assert result.matches is True


def test_non_numeric_options_not_applicable():
    result = verify_numeric_mcq("What is the SI unit of force?", ["Newton", "Joule", "Watt", "Pascal"], 0)
    assert not result.applicable
    assert result.reason == "not_all_options_numeric"


def test_word_problem_without_extractable_equation_not_applicable():
    result = verify_numeric_mcq(
        "A particle starts at rest and accelerates at 2 m/s^2 for 5 s. What is its final velocity?",
        ["10", "5", "2", "20"],
        0,
    )
    assert not result.applicable


def test_quadratic_with_multiple_solutions():
    result = verify_numeric_mcq("Solve for x: x^2 - 5x + 6 = 0", ["2", "3", "1", "6"], 0)
    assert result.applicable
    assert result.matches is True  # 2 is one of the two valid roots
