"""Dependency-free, deterministic embedding provider used as the dev/test
default when no internet access to a model hub is available (this
repository's own sandbox included -- see sentence_transformers_provider.py
for why).

This is a hashed, L2-normalized bag-of-words vector (a classic sparse
lexical-overlap representation, sometimes called "feature hashing" /
the "hashing trick"), NOT a learned semantic embedding model. It captures
word-overlap similarity, not paraphrase/synonym similarity -- good enough
to exercise the chunking -> embed -> vector-store -> retrieval pipeline's
plumbing and metadata filtering deterministically in tests, but it must
never be mistaken for a benchmarked semantic embedding model in
production. `GenerationModel`-style "never claim unmeasured numbers"
applies here too: this provider's retrieval quality is not evaluated
against docs/PHASE0-DESIGN.md section 6's embedding shortlist and isn't a
substitute for that evaluation.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .base import EmbeddingProviderMetadata, IEmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_index(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dimension


class HashingEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def metadata(self) -> EmbeddingProviderMetadata:
        return EmbeddingProviderMetadata(
            provider_type="hashing", model_name=f"hashing-{self.dimension}d", dimension=self.dimension
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(_tokenize(text))
        for token, count in counts.items():
            idx = _hash_index(token, self.dimension)
            vector[idx] += count
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
