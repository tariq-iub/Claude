"""Semantic deduplication (docs/PHASE0-DESIGN.md section 12, master
prompt section 23): 4 levels, checked cheapest-first so an obvious
duplicate never pays for an LLM call.

  1. Exact normalized-text hash.
  2. Lexical similarity (difflib ratio -- a dependency-free stand-in for
     a trigram/Levenshtein comparison, since the standard library already
     provides an adequate sequence-similarity ratio for this purpose).
  3. Embedding cosine similarity against candidates already retained for
     the same subject/topic.
  4. LLM-judge tie-break, ONLY for the borderline embedding-similarity
     band -- paying for an LLM call on every comparison would be wasteful
     when levels 1-3 already resolve the clear-cut cases.

Any existing candidate this new one is compared against is assumed
already-normalized/alive (callers filter to non-rejected candidates
before calling this module) -- dedup.py itself has no database
dependency, so it can be unit-tested without one.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import re

from backend.embeddings.base import IEmbeddingProvider
from backend.embeddings.similarity import cosine_similarity
from llm.providers.base import ILLMProvider

LEXICAL_SIMILARITY_THRESHOLD = 0.90
# Below this, embedding similarity is confidently "different questions";
# above BORDERLINE_HIGH, confidently "duplicate" -- only the band between
# the two triggers the more expensive LLM-judge tie-break.
BORDERLINE_LOW = 0.80
BORDERLINE_HIGH = 0.92

DUPLICATE_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["is_duplicate", "confidence", "reason"],
    "additionalProperties": False,
    "properties": {
        "is_duplicate": {
            "type": "boolean",
            "description": "True if the two questions test the same fact/concept, even if paraphrased differently.",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 4},
    },
}


@dataclasses.dataclass
class ExistingCandidate:
    id: int
    question_stem: str


@dataclasses.dataclass
class DedupResult:
    is_duplicate: bool
    method: str | None = None  # "exact_hash" | "lexical" | "embedding" | "llm_judge" | None
    matched_candidate_id: int | None = None
    score: float | None = None
    reason: str = ""


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def check_duplicate(
    question_stem: str,
    existing_candidates: list[ExistingCandidate],
    *,
    embedding_provider: IEmbeddingProvider | None = None,
    llm_provider: ILLMProvider | None = None,
) -> DedupResult:
    if not existing_candidates:
        return DedupResult(is_duplicate=False)

    target_hash = _exact_hash(question_stem)
    target_normalized = normalize_text(question_stem)

    # Level 1: exact hash.
    for existing in existing_candidates:
        if _exact_hash(existing.question_stem) == target_hash:
            return DedupResult(True, "exact_hash", existing.id, 1.0, "identical normalized text")

    # Level 2: lexical similarity.
    for existing in existing_candidates:
        ratio = difflib.SequenceMatcher(None, target_normalized, normalize_text(existing.question_stem)).ratio()
        if ratio >= LEXICAL_SIMILARITY_THRESHOLD:
            return DedupResult(True, "lexical", existing.id, ratio, f"lexical similarity {ratio:.3f}")

    # Level 3: embedding similarity (only if an embedding provider was supplied).
    if embedding_provider is not None:
        target_vector = embedding_provider.embed_query(question_stem)
        existing_vectors = embedding_provider.embed_texts([e.question_stem for e in existing_candidates])

        best_score = -1.0
        best_existing: ExistingCandidate | None = None
        for existing, vector in zip(existing_candidates, existing_vectors):
            sim = cosine_similarity(target_vector, vector)
            if sim > best_score:
                best_score = sim
                best_existing = existing

        if best_existing is not None and best_score >= BORDERLINE_HIGH:
            return DedupResult(True, "embedding", best_existing.id, best_score, f"embedding similarity {best_score:.3f}")

        if best_existing is not None and BORDERLINE_LOW <= best_score < BORDERLINE_HIGH and llm_provider is not None:
            # Level 4: LLM-judge tie-break, only for the borderline band.
            judge_result = _llm_judge_duplicate(llm_provider, question_stem, best_existing.question_stem)
            if judge_result is not None and judge_result["is_duplicate"]:
                return DedupResult(
                    True, "llm_judge", best_existing.id, best_score, judge_result.get("reason", "")
                )

    return DedupResult(is_duplicate=False)


def _llm_judge_duplicate(provider: ILLMProvider, question_a: str, question_b: str) -> dict | None:
    prompt = (
        f"Question A: {question_a}\n"
        f"Question B: {question_b}\n\n"
        "Do these two questions test the same underlying fact or concept, "
        "even if worded differently (e.g. 'What is the SI unit of force?' and "
        "'Force is measured in which SI unit?' test the same fact and ARE "
        "duplicates for question-bank diversity purposes)? Two questions about "
        "the same topic but testing different facts/aspects are NOT duplicates."
    )
    result = provider.generate_structured(prompt, json_schema=DUPLICATE_JUDGE_SCHEMA, temperature=0.1, max_tokens=200)
    return _try_parse_json(result.text)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
