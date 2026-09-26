from backend.validation.units import check_numeric_option_has_unit, has_recognized_unit, looks_numeric


def test_looks_numeric_true_for_plain_numbers():
    assert looks_numeric("42")
    assert looks_numeric("-9.8")
    assert looks_numeric("3.0e8")


def test_looks_numeric_false_for_text():
    assert not looks_numeric("Newton")
    assert not looks_numeric("12 m/s")  # has trailing text, not a bare number


def test_has_recognized_unit_detects_common_si_units():
    assert has_recognized_unit("m/s")
    assert has_recognized_unit("kg")
    assert has_recognized_unit("12 N")
    assert has_recognized_unit("5 J")


def test_has_recognized_unit_detects_prefixed_units():
    assert has_recognized_unit("5 km")
    assert has_recognized_unit("3 ms")
    assert has_recognized_unit("2 kJ")


def test_numeric_option_with_unit_returns_true():
    assert check_numeric_option_has_unit("9.8 m/s^2") is True
    assert check_numeric_option_has_unit("5 kg") is True


def test_numeric_option_without_unit_returns_false():
    assert check_numeric_option_has_unit("4.9") is False
    assert check_numeric_option_has_unit("-273") is False


def test_non_numeric_option_returns_none():
    assert check_numeric_option_has_unit("Newton's second law") is None
    assert check_numeric_option_has_unit("Force") is None


def test_scientific_notation_with_unit_returns_true():
    assert check_numeric_option_has_unit("3.0 x 10^8 m/s") is True


def test_percentage_and_angle_units_recognized():
    assert check_numeric_option_has_unit("30 deg") is True
    assert check_numeric_option_has_unit("45°") is True
    assert check_numeric_option_has_unit("50%") is True
