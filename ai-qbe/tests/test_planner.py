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


def test_build_plan_defaults_question_type_to_single_best_answer():
    plan = build_plan({1: 5}, {"easy": 1.0}, {"remember": 1.0})
    assert all(cell.question_type == "single_best_answer" for cell in plan)


def test_build_plan_distributes_question_types():
    topic_targets = {1: 100}
    difficulty_distribution = {"easy": 1.0}
    bloom_distribution = {"remember": 1.0}
    question_type_distribution = {
        "single_best_answer": 0.6,
        "scenario_based": 0.25,
        "negative": 0.10,
        "definition_recall": 0.05,
    }

    plan = build_plan(topic_targets, difficulty_distribution, bloom_distribution, question_type_distribution)

    totals_by_type = {}
    for cell in plan:
        totals_by_type[cell.question_type] = totals_by_type.get(cell.question_type, 0) + cell.count
    assert sum(totals_by_type.values()) == 100
    assert set(totals_by_type.keys()).issubset(set(question_type_distribution.keys()))
    # single_best_answer should dominate given its 0.6 share
    assert totals_by_type.get("single_best_answer", 0) > totals_by_type.get("negative", 0)


def test_build_plan_full_three_way_split_sums_to_topic_target():
    topic_targets = {1: 47}  # deliberately awkward number to stress largest-remainder math
    difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    bloom_distribution = {"remember": 0.25, "understand": 0.3, "apply": 0.3, "analyze": 0.15}
    question_type_distribution = {"single_best_answer": 0.6, "scenario_based": 0.4}

    plan = build_plan(topic_targets, difficulty_distribution, bloom_distribution, question_type_distribution)
    assert sum(cell.count for cell in plan) == 47
