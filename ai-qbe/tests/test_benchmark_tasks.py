"""Sanity checks on the Phase 1 benchmark task set itself.

Confirms the ~100-task set (master prompt Phase 1: "create approximately
100 representative generation/verification tasks") is well-formed and
that the ground truth baked into verification_tasks.json /
math_computation_tasks.json is actually correct -- these are the fixtures
every model's score gets compared against, so a wrong fixture would
silently invalidate every benchmark run.
"""

import json
import sys
from pathlib import Path

import sympy

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "scripts" / "benchmark" / "tasks"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))


def load(name):
    return json.loads((TASKS_DIR / name).read_text())


def test_total_task_count_is_approximately_100():
    total = (
        len(load("mcq_generation_tasks.json"))
        + len(load("verification_tasks.json"))
        + len(load("json_stress_tasks.json"))
        + len(load("math_computation_tasks.json"))
    )
    assert total == 100


def test_generation_tasks_have_required_fields_and_unique_ids():
    tasks = load("mcq_generation_tasks.json")
    ids = set()
    for t in tasks:
        for field in ("id", "domain", "topic", "bloom_level", "difficulty", "context", "instruction"):
            assert field in t, f"missing {field} in {t.get('id')}"
        assert t["bloom_level"] in ("remember", "understand", "apply", "analyze")
        assert t["difficulty"] in ("easy", "medium", "hard")
        ids.add(t["id"])
    assert len(ids) == len(tasks)


def test_generation_tasks_cover_math_physics_chemistry():
    tasks = load("mcq_generation_tasks.json")
    domains = {t["domain"] for t in tasks}
    assert {"math", "physics", "chemistry"}.issubset(domains)


def test_verification_tasks_ground_truth_is_internally_consistent():
    tasks = load("verification_tasks.json")
    for t in tasks:
        claimed = t["claimed_correct_option"]
        truth = t["ground_truth_correct_option"]
        if claimed == truth:
            assert t["expected_verdict"] == "PASS", t["id"]
        else:
            assert t["expected_verdict"] == "FAIL", t["id"]


def test_verification_tasks_options_are_indexable():
    tasks = load("verification_tasks.json")
    for t in tasks:
        assert 0 <= t["ground_truth_correct_option"] < len(t["options"])
        assert 0 <= t["claimed_correct_option"] < len(t["options"])


def test_math_computation_expected_values_are_actually_correct():
    """Independently re-derive each expected_sympy value with SymPy,
    rather than trusting the hand-authored fixture."""
    tasks = load("math_computation_tasks.json")
    x, y = sympy.symbols("x y")

    for t in tasks:
        op = t["sympy_op"]
        if op == "diff":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.diff(expr, x)
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "integrate":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.integrate(expr, x)
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "definite_integrate":
            expr = sympy.sympify(t["sympy_expr"])
            lo, hi = t["bounds"]
            derived = sympy.integrate(expr, (x, lo, hi))
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "simplify":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.simplify(expr)
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "expand":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.expand(expr)
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "factor":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.factor(expr)
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "limit":
            expr = sympy.sympify(t["sympy_expr"])
            derived = sympy.limit(expr, x, t["limit_point"])
            expected = sympy.sympify(t["expected_sympy"])
            assert sympy.simplify(derived - expected) == 0, t["id"]
        elif op == "solve":
            expr = sympy.sympify(t["sympy_expr"])
            derived = set(sympy.solve(expr, x))
            expected = set(sympy.sympify(v) for v in json.loads(t["expected_sympy"]))
            assert derived == expected, t["id"]
        elif op == "linsolve":
            eqs = [sympy.sympify(e) for e in t["system"]]
            vars_ = sympy.symbols(t["sympy_vars"])
            derived = sympy.linsolve(eqs, vars_)
            expected = sympy.sympify(t["expected_sympy"])
            assert derived == expected, t["id"]
        else:
            raise AssertionError(f"unknown op {op} in {t['id']}")


def test_json_stress_tasks_have_prompts():
    tasks = load("json_stress_tasks.json")
    for t in tasks:
        assert t["prompt"].strip()
        assert t["id"]
