"""Distractor quality validation (docs/PHASE0-DESIGN.md section 12,
master prompt section 14).

Exact-duplicate options and raw option-length limits are already caught
by structural validation (backend/validation/structural.py) before this
runs -- this module covers what structural validation can't: near-
duplicate (paraphrased) options, an option suspiciously close to the
correct answer (a possible second correct option), and length-outlier
distractors that could tip off the answer through pattern alone.

Embedding-based checks are best-effort: if no `IEmbeddingProvider` is
supplied (e.g. a manual-context job with no RAG infrastructure wired up),
only the embedding-independent checks run, and `embedding_checked=False`
is set so callers know exactly what wasn't checked.
"""

from __future__ import annotations

import dataclasses

from backend.embeddings.base import IEmbeddingProvider
from backend.embeddings.similarity import cosine_similarity

# Above this cosine similarity, two distinct options are considered
# near-duplicate paraphrases of each other rather than genuinely
# different distractors.
NEAR_DUPLICATE_THRESHOLD = 0.92
# Above this similarity between the correct option and an incorrect one,
# the incorrect option might also be arguably correct (a "multiple
# correct answers" risk) and is flagged for human attention.
POSSIBLE_MULTIPLE_CORRECT_THRESHOLD = 0.90
# An option whose word count deviates this much from the median across
# the option set is flagged as a length outlier (a common test-taking
# "the longest/most-specific option is usually correct" tell).
LENGTH_OUTLIER_RATIO = 2.0


@dataclasses.dataclass
class DistractorValidationResult:
    passed: bool
    embedding_checked: bool
    reasons: list[str] = dataclasses.field(default_factory=list)
    flags: list[str] = dataclasses.field(default_factory=list)
    score: float = 100.0  # 0-100, reduced per soft flag; reasons drive `passed`


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return float(ordered[mid])


def validate_distractors(
    options: list[str],
    correct_option: int,
    *,
    embedding_provider: IEmbeddingProvider | None = None,
) -> DistractorValidationResult:
    reasons: list[str] = []
    flags: list[str] = []
    score = 100.0

    word_counts = [len(opt.split()) for opt in options]
    median_len = _median(word_counts)
    if median_len > 0:
        for i, count in enumerate(word_counts):
            if count == 0:
                continue
            ratio = max(count / median_len, median_len / count) if count else float("inf")
            if ratio >= LENGTH_OUTLIER_RATIO:
                flags.append(f"option_{i}_length_outlier (words={count}, median={median_len})")
                score -= 10

    embedding_checked = embedding_provider is not None
    if embedding_provider is not None:
        vectors = embedding_provider.embed_texts(options)
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim >= NEAR_DUPLICATE_THRESHOLD:
                    reasons.append(f"options_{i}_and_{j}_near_duplicate (similarity={sim:.3f})")

        for i, vector in enumerate(vectors):
            if i == correct_option:
                continue
            sim_to_correct = cosine_similarity(vectors[correct_option], vector)
            if sim_to_correct >= POSSIBLE_MULTIPLE_CORRECT_THRESHOLD:
                flags.append(f"option_{i}_suspiciously_close_to_correct_answer (similarity={sim_to_correct:.3f})")
                score -= 15

    return DistractorValidationResult(
        passed=len(reasons) == 0,
        embedding_checked=embedding_checked,
        reasons=reasons,
        flags=flags,
        score=max(0.0, score),
    )
