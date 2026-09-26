"""MCQ Generation Planner (docs/PHASE0-DESIGN.md section 11 / master
prompt section 9).

Full topic x difficulty x Bloom x question-type matrix (Phase 5). Each
`PlanCell.count` is a target number of questions for one specific
combination; the executor's batch generator (docs/PHASE0-DESIGN.md
section 11: "10-30 questions per call") then requests that many items
from the LLM in as few calls as practical for the cell, rather than one
call per question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_DEFAULT_QUESTION_TYPE_DISTRIBUTION = {"single_best_answer": 1.0}


@dataclass
class PlanCell:
    generation_job_topic_id: int
    bloom_level: str
    difficulty: str
    count: int
    question_type: str = "single_best_answer"


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
    (e.g. difficulty_distribution / bloom_distribution / question_type
    distribution on the job), again using largest-remainder apportionment.
    """
    return allocate_topic_counts(count, distribution)  # same math, different key type


def build_plan(
    topic_target_counts: dict[int, int],
    difficulty_distribution: dict[str, float],
    bloom_distribution: dict[str, float],
    question_type_distribution: dict[str, float] | None = None,
) -> list[PlanCell]:
    """Cross each topic's target count with the job's difficulty, Bloom,
    and question-type distributions to produce concrete generation cells.

    `question_type_distribution` defaults to 100% single_best_answer,
    which reproduces the pre-Phase-5 two-way (difficulty x Bloom) plan
    exactly -- existing callers that don't pass it are unaffected.
    """
    question_type_distribution = question_type_distribution or _DEFAULT_QUESTION_TYPE_DISTRIBUTION

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
                by_type = allocate_distribution(bloom_count, question_type_distribution)
                for question_type, type_count in by_type.items():
                    if type_count == 0:
                        continue
                    plan.append(PlanCell(topic_id, bloom_level, difficulty, type_count, question_type))
    return plan
