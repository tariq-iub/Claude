from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from backend.validation.distractor import validate_distractors


def test_clean_options_pass_without_embedding_provider():
    result = validate_distractors(["Newton", "Joule", "Watt", "Pascal"], 0)
    assert result.passed
    assert not result.embedding_checked
    assert result.reasons == []


def test_length_outlier_flagged_as_soft_signal_not_a_failure():
    options = ["A", "B", "C", "This option is much longer than all the others by a wide margin"]
    result = validate_distractors(options, 0)
    assert result.passed  # soft flag, not a hard failure
    assert any("length_outlier" in f for f in result.flags)
    assert result.score < 100.0


def test_near_duplicate_options_detected_with_embedding_provider():
    embedder = HashingEmbeddingProvider(dimension=256)
    options = [
        "Newton's second law of motion force mass acceleration",
        "Newton's second law of motion force mass acceleration exactly",
        "Conservation of energy principle",
        "Ohm's law relates voltage current resistance",
    ]
    result = validate_distractors(options, 0, embedding_provider=embedder)
    assert not result.passed
    assert any("near_duplicate" in r for r in result.reasons)


def test_option_suspiciously_close_to_correct_is_flagged():
    embedder = HashingEmbeddingProvider(dimension=256)
    options = [
        "Force equals mass times acceleration in classical mechanics always exactly",
        "Force equals mass times acceleration in classical mechanics",
        "Energy is conserved in closed systems",
        "Momentum is conserved during collisions",
    ]
    result = validate_distractors(options, 0, embedding_provider=embedder)
    # option 1 is textually very close to the correct option 0 (similarity
    # ~0.91: high enough to flag as a possible second-correct-answer risk,
    # but below the near-duplicate threshold, so it's a soft flag here
    # rather than a hard-failing "reason").
    assert result.passed
    assert any("suspiciously_close_to_correct_answer" in f for f in result.flags)


def test_distinct_options_with_embedding_provider_pass_cleanly():
    embedder = HashingEmbeddingProvider(dimension=256)
    options = ["Newton's second law", "Conservation of energy", "Ohm's law", "Boyle's law"]
    result = validate_distractors(options, 0, embedding_provider=embedder)
    assert result.passed
    assert result.embedding_checked


def test_embedding_checked_flag_reflects_whether_provider_was_supplied():
    result_without = validate_distractors(["A", "B", "C", "D"], 0)
    assert not result_without.embedding_checked

    result_with = validate_distractors(["A", "B", "C", "D"], 0, embedding_provider=HashingEmbeddingProvider())
    assert result_with.embedding_checked
