import math

from backend.embeddings.hashing_provider import HashingEmbeddingProvider


def test_embed_texts_returns_correct_dimension():
    provider = HashingEmbeddingProvider(dimension=128)
    vectors = provider.embed_texts(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert all(len(v) == 128 for v in vectors)


def test_embed_is_deterministic():
    provider = HashingEmbeddingProvider(dimension=128)
    v1 = provider.embed_query("Newton's second law of motion")
    v2 = provider.embed_query("Newton's second law of motion")
    assert v1 == v2


def test_embed_vectors_are_unit_normalized():
    provider = HashingEmbeddingProvider(dimension=128)
    vector = provider.embed_query("force equals mass times acceleration")
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6


def test_shared_vocabulary_texts_are_more_similar_than_unrelated_ones():
    provider = HashingEmbeddingProvider(dimension=256)

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    physics_a = provider.embed_query("Newton's second law relates force mass and acceleration")
    physics_b = provider.embed_query("Force equals mass times acceleration in Newtonian mechanics")
    chemistry = provider.embed_query("Balancing chemical equations requires equal atoms on both sides")

    sim_related = cosine(physics_a, physics_b)
    sim_unrelated = cosine(physics_a, chemistry)
    assert sim_related > sim_unrelated


def test_empty_text_produces_zero_vector_without_error():
    provider = HashingEmbeddingProvider(dimension=64)
    vector = provider.embed_query("")
    assert vector == [0.0] * 64


def test_metadata_reports_configured_dimension():
    provider = HashingEmbeddingProvider(dimension=64)
    meta = provider.metadata()
    assert meta.dimension == 64
    assert meta.provider_type == "hashing"
