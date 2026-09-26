"""Vector similarity helper shared by the Phase 7 QA pipeline (distractor
validation, semantic deduplication) -- kept next to the embedding
providers since both vectors it compares always come from the same
`IEmbeddingProvider`, and correctness here depends on knowing whether
that provider's vectors are pre-normalized.
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Assumes neither vector is all-zero; callers pre-filter empty text.
    Does NOT assume the vectors are pre-normalized (computes true cosine),
    so it works correctly regardless of which `IEmbeddingProvider`
    produced them.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
