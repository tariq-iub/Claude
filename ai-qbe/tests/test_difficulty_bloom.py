from backend.domain.enums import BloomLevel, DifficultyLevel
from backend.validation.difficulty_bloom import check_difficulty_bloom, estimate_bloom, estimate_difficulty


def test_estimate_bloom_remember_keywords():
    assert estimate_bloom("What is the SI unit of force?") == BloomLevel.REMEMBER
    assert estimate_bloom("Define kinetic energy.") == BloomLevel.REMEMBER


def test_estimate_bloom_apply_keywords():
    assert estimate_bloom("Calculate the force given mass and acceleration.") == BloomLevel.APPLY


def test_estimate_bloom_analyze_keywords():
    assert estimate_bloom("Compare the effects of increasing mass versus increasing velocity.") == BloomLevel.ANALYZE


def test_estimate_bloom_none_when_no_keywords_match():
    assert estimate_bloom("Force equals mass times acceleration.") is None


def test_estimate_difficulty_easy_for_simple_conceptual_question():
    result = estimate_difficulty("What is the SI unit of force?", ["Newton", "Joule", "Watt", "Pascal"])
    assert result == DifficultyLevel.EASY


def test_estimate_difficulty_higher_for_calculation_with_numeric_options():
    result = estimate_difficulty(
        "A 2 kg mass accelerates at 3 m/s^2. What is the net force?", ["6", "5", "9", "1"]
    )
    assert result in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD)


def test_estimate_difficulty_hard_for_multi_sentence_calculation_scenario():
    stem = (
        "A particle starts at rest. It accelerates at 2 m/s^2 for 5 seconds. "
        "What is its final velocity?"
    )
    result = estimate_difficulty(stem, ["10", "5", "2", "20"])
    assert result == DifficultyLevel.HARD


def test_check_difficulty_bloom_flags_mismatch():
    result = check_difficulty_bloom(
        "Calculate the resulting force.",
        ["6 N", "5 N", "9 N", "1 N"],
        declared_bloom=BloomLevel.REMEMBER,
        declared_difficulty=DifficultyLevel.EASY,
    )
    assert result.bloom_match is False
    assert result.notes


def test_check_difficulty_bloom_no_mismatch_when_aligned():
    result = check_difficulty_bloom(
        "What is the SI unit of force?",
        ["Newton", "Joule", "Watt", "Pascal"],
        declared_bloom=BloomLevel.REMEMBER,
        declared_difficulty=DifficultyLevel.EASY,
    )
    assert result.bloom_match is True
    assert result.difficulty_match is True


def test_check_difficulty_bloom_none_when_no_keyword_estimate():
    result = check_difficulty_bloom(
        "Force equals mass times acceleration.",
        ["A", "B", "C", "D"],
        declared_bloom=BloomLevel.UNDERSTAND,
        declared_difficulty=DifficultyLevel.EASY,
    )
    assert result.bloom_match is None  # no confident estimate, not treated as a mismatch
