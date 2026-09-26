from __future__ import annotations

from .base import IEmbeddingProvider
from .hashing_provider import HashingEmbeddingProvider


def build_embedding_provider(provider_type: str, **kwargs) -> IEmbeddingProvider:
    if provider_type == "hashing":
        return HashingEmbeddingProvider(**kwargs)
    if provider_type == "sentence_transformers":
        from .sentence_transformers_provider import SentenceTransformersEmbeddingProvider

        return SentenceTransformersEmbeddingProvider(**kwargs)
    raise ValueError(f"Unknown embedding provider_type: {provider_type}")
