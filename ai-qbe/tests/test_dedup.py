from backend.embeddings.hashing_provider import HashingEmbeddingProvider
from backend.validation.dedup import ExistingCandidate, check_duplicate, normalize_text
from llm.providers.mock_provider import MockProvider


def test_no_existing_candidates_is_never_a_duplicate():
    result = check_duplicate("What is the SI unit of force?", [])
    assert not result.is_duplicate


def test_exact_normalized_text_match_is_level1_duplicate():
    existing = [ExistingCandidate(id=1, question_stem="What is the SI unit of force?")]
    result = check_duplicate("what is the SI unit of force?!", existing)
    assert result.is_duplicate
    assert result.method == "exact_hash"
    assert result.matched_candidate_id == 1


def test_minor_wording_variation_is_level2_lexical_duplicate():
    existing = [ExistingCandidate(id=2, question_stem="What is the SI unit of force in physics?")]
    result = check_duplicate("What is the SI unit of force in Physics?", existing)
    assert result.is_duplicate
    assert result.method in ("exact_hash", "lexical")


def test_clearly_different_questions_are_not_duplicates():
    existing = [ExistingCandidate(id=3, question_stem="What is the SI unit of force?")]
    result = check_duplicate("Balance the equation: H2 + O2 -> H2O", existing)
    assert not result.is_duplicate


def test_high_word_overlap_is_embedding_level_duplicate():
    embedder = HashingEmbeddingProvider(dimension=256)
    existing = [
        ExistingCandidate(
            id=4,
            question_stem="Newton's second law relates force mass and acceleration in classical mechanics",
        )
    ]
    result = check_duplicate(
        "Newton's second law relates force mass and acceleration in classical dynamics",
        existing,
        embedding_provider=embedder,
    )
    assert result.is_duplicate
    assert result.method in ("lexical", "embedding")


def test_embedding_none_skips_level3_check_gracefully():
    existing = [ExistingCandidate(id=5, question_stem="A completely unrelated question about chemistry bonding")]
    result = check_duplicate("A totally different question about vectors", existing, embedding_provider=None)
    assert not result.is_duplicate


def test_llm_judge_only_invoked_for_borderline_band(monkeypatch):
    embedder = HashingEmbeddingProvider(dimension=256)
    provider = MockProvider()

    calls = []
    original = provider.generate_structured

    def spy(prompt, *, json_schema, **kwargs):
        calls.append(json_schema)
        return original(prompt, json_schema=json_schema, **kwargs)

    monkeypatch.setattr(provider, "generate_structured", spy)

    # Clearly different -> no LLM call needed at all.
    existing = [ExistingCandidate(id=6, question_stem="Balance the chemical equation for combustion of methane")]
    check_duplicate("What is the derivative of x squared", existing, embedding_provider=embedder, llm_provider=provider)
    assert len(calls) == 0


def test_llm_judge_is_invoked_for_borderline_similarity_and_its_verdict_is_used(monkeypatch):
    """Similarity ~0.825 falls inside the borderline band, so the LLM
    judge must actually be called (unlike the clearly-different case
    above), and a judge verdict of "is_duplicate: true" must be honored."""
    embedder = HashingEmbeddingProvider(dimension=256)

    class JudgeSaysD(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            if "is_duplicate" in json_schema.get("properties", {}):
                import json as _json

                from llm.providers.base import GenerationResult

                obj = {"is_duplicate": True, "confidence": 0.9, "reason": "Both test the mole concept."}
                return GenerationResult(_json.dumps(obj), 10, 10, 0.01)
            return super().generate_structured(prompt, json_schema=json_schema, **kwargs)

    existing = [ExistingCandidate(id=7, question_stem="What is the mole concept used for in chemistry")]
    result = check_duplicate(
        "What is the mole concept applied in chemistry",
        existing,
        embedding_provider=embedder,
        llm_provider=JudgeSaysD(),
    )
    assert result.is_duplicate
    assert result.method == "llm_judge"
    assert result.matched_candidate_id == 7


def test_llm_judge_verdict_of_not_duplicate_is_also_honored():
    embedder = HashingEmbeddingProvider(dimension=256)

    class JudgeSaysNotD(MockProvider):
        def generate_structured(self, prompt, *, json_schema, **kwargs):
            if "is_duplicate" in json_schema.get("properties", {}):
                import json as _json

                from llm.providers.base import GenerationResult

                obj = {"is_duplicate": False, "confidence": 0.8, "reason": "Different aspect of the same topic."}
                return GenerationResult(_json.dumps(obj), 10, 10, 0.01)
            return super().generate_structured(prompt, json_schema=json_schema, **kwargs)

    existing = [ExistingCandidate(id=8, question_stem="What is the mole concept used for in chemistry")]
    result = check_duplicate(
        "What is the mole concept applied in chemistry",
        existing,
        embedding_provider=embedder,
        llm_provider=JudgeSaysNotD(),
    )
    assert not result.is_duplicate


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("What Is Force?!") == normalize_text("what is force")
