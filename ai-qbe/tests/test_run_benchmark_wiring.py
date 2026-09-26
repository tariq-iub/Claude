"""End-to-end wiring test for run_benchmark.py using a fake ILLMProvider.

This proves the harness's plumbing (task loading -> provider call ->
grading -> report assembly) is correct without needing a real GPU/model --
that part still requires the target workstation (see
scripts/benchmark/README.md). A fake provider returns pre-baked,
deliberately mixed correct/incorrect responses so we can assert the
summary numbers come out as expected.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from llm.providers.base import GenerationResult, ILLMProvider, ProviderMetadata  # noqa: E402

import run_benchmark  # noqa: E402


class FakeProvider(ILLMProvider):
    """Always returns a valid, correct MCQ / verification / math answer,
    with a synthetic 0.01s latency and 42 tokens, so tests are fast and
    deterministic."""

    def metadata(self):
        return ProviderMetadata("fake", "fake-model", "v1", "none", 4096)

    def health_check(self):
        return True

    def generate_text(self, prompt, **kwargs):
        return GenerationResult("{}", 10, 42, 0.01)

    def generate_structured(self, prompt, *, json_schema, **kwargs):
        if "answer" in json_schema.get("properties", {}) and len(json_schema["properties"]) == 1:
            # math_computation task
            return GenerationResult(json.dumps({"answer": "0"}), 10, 5, 0.01)
        if "verdict" in json_schema.get("properties", {}):
            return GenerationResult(
                json.dumps({"verdict": "PASS", "confidence": 0.9, "reason": "matches"}), 10, 20, 0.01
            )
        mcq = {
            "question": "Placeholder question?",
            "options": ["A", "B", "C", "D"],
            "correct_option": 0,
            "explanation": "Because A.",
            "topic": "t",
            "difficulty": "easy",
            "bloom_level": "remember",
            "sources": [],
            "confidence": 0.9,
        }
        return GenerationResult(json.dumps(mcq), 10, 30, 0.01)


def test_run_generation_tasks_all_structurally_valid():
    provider = FakeProvider()
    schema = json.loads((REPO_ROOT / "llm" / "schemas" / "mcq_schema.json").read_text())
    results = run_benchmark.run_generation_tasks(provider, schema)
    assert len(results) == 40
    assert all(r["structural_pass"] for r in results)


def test_run_verification_tasks_structurally_valid_but_accuracy_varies():
    provider = FakeProvider()
    schema = json.loads((REPO_ROOT / "llm" / "schemas" / "verification_schema.json").read_text())
    results = run_benchmark.run_verification_tasks(provider, schema)
    assert len(results) == 30
    assert all(r["structural_pass"] for r in results)
    # FakeProvider always says PASS; exactly the tasks whose expected_verdict
    # is PASS should be scored correct, the FAIL ones should be scored wrong.
    tasks = run_benchmark.load_json(run_benchmark.TASKS_DIR / "verification_tasks.json")
    expected_pass_count = sum(1 for t in tasks if t["expected_verdict"] == "PASS")
    actually_correct = sum(1 for r in results if r["verification_accuracy"]["passed"])
    assert actually_correct == expected_pass_count


def test_run_math_computation_tasks_reports_incorrect_for_wrong_fake_answer():
    provider = FakeProvider()
    results = run_benchmark.run_math_computation_tasks(provider)
    assert len(results) == 10
    # FakeProvider always answers "0", which is wrong for every one of our
    # 10 real math tasks -- this should show up as 0% correctness, proving
    # the harness doesn't just trust structural validity as correctness.
    assert all(not r["correct"] for r in results)


def test_summarize_computes_rates_correctly():
    rows = [{"structural_pass": True}, {"structural_pass": False}, {"structural_pass": True}]
    summary = run_benchmark.summarize(rows, "structural_pass")
    assert summary["total"] == 3
    assert summary["pass_rate"] == 2 / 3
