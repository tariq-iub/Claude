from backend.validation.latex_extraction import extract_math_segments, has_math_notation


def test_extracts_inline_dollar_math():
    segments = extract_math_segments("The equation $F=ma$ is fundamental.")
    assert len(segments) == 1
    assert segments[0].latex == "F=ma"
    assert segments[0].display is False


def test_extracts_display_double_dollar_math():
    segments = extract_math_segments("Consider: $$\\int_0^1 x^2\\,dx$$ as an example.")
    assert len(segments) == 1
    assert segments[0].latex == "\\int_0^1 x^2\\,dx"
    assert segments[0].display is True


def test_extracts_bracket_display_math():
    segments = extract_math_segments("Result: \\[E=mc^2\\] shown above.")
    assert len(segments) == 1
    assert segments[0].latex == "E=mc^2"
    assert segments[0].display is True


def test_extracts_paren_inline_math():
    segments = extract_math_segments("We have \\(x^2+y^2=r^2\\) as the circle equation.")
    assert len(segments) == 1
    assert segments[0].latex == "x^2+y^2=r^2"
    assert segments[0].display is False


def test_extracts_multiple_segments_in_order():
    text = "First $a=1$, then $b=2$, and finally $c=3$."
    segments = extract_math_segments(text)
    assert [s.latex for s in segments] == ["a=1", "b=2", "c=3"]


def test_double_dollar_not_misparsed_as_two_empty_inline_pairs():
    text = "$$x^2$$"
    segments = extract_math_segments(text)
    assert len(segments) == 1
    assert segments[0].display is True
    assert segments[0].latex == "x^2"


def test_no_math_returns_empty_list():
    segments = extract_math_segments("Plain text with no notation at all.")
    assert segments == []
    assert not has_math_notation("Plain text with no notation at all.")


def test_has_math_notation_true_when_present():
    assert has_math_notation("The formula $E=mc^2$ is famous.")


def test_chemistry_ce_wrapped_in_dollar_signs_is_extracted():
    text = "The reaction $\\ce{2H2 + O2 -> 2H2O}$ produces water."
    segments = extract_math_segments(text)
    assert len(segments) == 1
    assert segments[0].latex == "\\ce{2H2 + O2 -> 2H2O}"
