from backend.generation.planner import allocate_topic_counts, build_plan


def test_allocate_topic_counts_sums_to_exact_requested_count():
    weights = {1: 1.0, 2: 1.0, 3: 1.0}
    result = allocate_topic_counts(10, weights)
    assert sum(result.values()) == 10


def test_allocate_topic_counts_proportional_to_weight():
    weights = {1: 3.0, 2: 1.0}
    result = allocate_topic_counts(100, weights)
    # topic 1 should get roughly 3x topic 2's share
    assert result[1] > result[2]
    assert abs(result[1] - 75) <= 1
    assert abs(result[2] - 25) <= 1


def test_allocate_topic_counts_uneven_division_still_sums_correctly():
    weights = {1: 1.0, 2: 1.0, 3: 1.0}
    result = allocate_topic_counts(10, weights)  # 10/3 doesn't divide evenly
    assert sum(result.values()) == 10
    assert max(result.values()) - min(result.values()) <= 1


def test_allocate_topic_counts_empty_weights_returns_empty():
    assert allocate_topic_counts(10, {}) == {}


def test_allocate_topic_counts_rejects_zero_total_weight():
    import pytest

    with pytest.raises(ValueError):
        allocate_topic_counts(10, {1: 0.0, 2: 0.0})


def test_build_plan_cells_sum_to_topic_targets():
    topic_targets = {1: 20, 2: 10}
    difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    bloom_distribution = {"remember": 0.25, "understand": 0.3, "apply": 0.3, "analyze": 0.15}

    plan = build_plan(topic_targets, difficulty_distribution, bloom_distribution)

    totals_by_topic = {}
    for cell in plan:
        totals_by_topic[cell.generation_job_topic_id] = (
            totals_by_topic.get(cell.generation_job_topic_id, 0) + cell.count
        )
    assert totals_by_topic == topic_targets


def test_build_plan_skips_zero_target_topics():
    topic_targets = {1: 0, 2: 5}
    plan = build_plan(topic_targets, {"easy": 1.0}, {"remember": 1.0})
    assert all(cell.generation_job_topic_id != 1 for cell in plan)
