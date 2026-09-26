"""MCQ Generation Planner (docs/PHASE0-DESIGN.md section 11 / master
prompt section 9).

Phase 2 scope: distribute a job's requested_count across its topics by
weight, and each topic's allocation across the job's configured
difficulty/Bloom distributions. Phase 5 extends this into a full
topic x subtopic x Bloom x difficulty x question-type matrix with batching
and an over-generation controller driven by observed rejection rates --
this module is the part of that planner that's needed before a single
candidate can be requested from an LLM at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PlanCell:
    generation_job_topic_id: int
    bloom_level: str
    difficulty: str
    count: int


def allocate_topic_counts(requested_count: int, topic_weights: dict[int, float]) -> dict[int, int]:
    """Distribute `requested_count` across topics proportional to weight.

    Uses largest-remainder apportionment so the per-topic counts always
    sum to exactly `requested_count` (simple round-then-truncate can drift
    by a few questions on large jobs, which would silently under- or
    over-deliver against what the administrator asked for).
    """
    if not topic_weights:
        return {}
    total_weight = sum(topic_weights.values())
    if total_weight <= 0:
        raise ValueError("topic_weights must sum to a positive number")

    exact = {
        topic_id: requested_count * (weight / total_weight)
        for topic_id, weight in topic_weights.items()
    }
    floors = {topic_id: math.floor(v) for topic_id, v in exact.items()}
    remainder = requested_count - sum(floors.values())

    remainders_sorted = sorted(exact.items(), key=lambda kv: kv[1] - floors[kv[0]], reverse=True)
    for topic_id, _ in remainders_sorted[:remainder]:
        floors[topic_id] += 1

    return floors


def allocate_distribution(count: int, distribution: dict[str, float]) -> dict[str, int]:
    """Split `count` items across a {label: fraction} distribution
    (e.g. difficulty_distribution / bloom_distribution on the job),
    again using largest-remainder apportionment.
    """
    return allocate_topic_counts(count, distribution)  # same math, different key type


def build_plan(
    topic_target_counts: dict[int, int],
    difficulty_distribution: dict[str, float],
    bloom_distribution: dict[str, float],
) -> list[PlanCell]:
    """Cross each topic's target count with the job's difficulty and Bloom
    distributions to produce concrete generation cells.

    Phase 2 keeps this a simple two-way split (per topic: difficulty counts,
    then each difficulty's count split again by Bloom) rather than a full
    joint distribution, since Phase 2 generation batches are small and
    manually scoped -- a joint topic x Bloom x difficulty x question-type
    matrix is Phase 5 scope once real batching/concurrency exists.
    """
    plan: list[PlanCell] = []
    for topic_id, target_count in topic_target_counts.items():
        if target_count == 0:
            continue
        by_difficulty = allocate_distribution(target_count, difficulty_distribution)
        for difficulty, diff_count in by_difficulty.items():
            if diff_count == 0:
                continue
            by_bloom = allocate_distribution(diff_count, bloom_distribution)
            for bloom_level, bloom_count in by_bloom.items():
                if bloom_count == 0:
                    continue
                plan.append(PlanCell(topic_id, bloom_level, difficulty, bloom_count))
    return plan
