#!/usr/bin/env python3
"""Phase 1 local-LLM benchmark harness.

Runs the ~100-task set (scripts/benchmark/tasks/*.json) against a
configured ILLMProvider and produces a JSON + Markdown report with:
  - JSON structural compliance rate (schema validity)
  - option-conciseness / distractor-hygiene pass rates
  - deterministic correctness (SymPy) for math_computation_tasks
  - verifier-verdict accuracy for verification_tasks
  - latency / tokens-per-second
  - GPU/RAM utilization sampled during the run (best-effort)

It does NOT score academic quality (clarity, distractor plausibility,
grounding, Bloom/difficulty accuracy) -- those require the expert rubric
in report_template.md, per the master prompt's ban on an LLM (or a
benchmark script) being the sole judge of its own quality.

Usage (run on the actual 8GB-GPU workstation, not in a sandbox without a
GPU/model runtime):

    python run_benchmark.py --provider ollama --model qwen2.5:7b-instruct-q4_K_M \\
        --context-window 4096 --output-dir ./results/qwen2.5-7b-q4

See README.md in this directory for full setup instructions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: ai-qbe/

from llm.providers import ILLMProvider, LlamaCppProvider, OllamaProvider, OpenAICompatibleProvider  # noqa: E402

from resource_monitor import ResourceMonitor  # noqa: E402
import validators  # noqa: E402

TASKS_DIR = Path(__file__).parent / "tasks"
SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "llm" / "schemas"


def build_provider(args) -> ILLMProvider:
    if args.provider == "ollama":
        return OllamaProvider(
            args.model,
            base_url=args.base_url or "http://localhost:11434",
            model_version=args.model_version,
            quantization=args.quantization,
            context_window=args.context_window,
        )
    if args.provider == "llamacpp":
        return LlamaCppProvider(
            args.model,
            base_url=args.base_url or "http://localhost:8080",
            model_version=args.model_version,
            quantization=args.quantization,
            context_window=args.context_window,
        )
    if args.provider == "openai_compatible":
        if not args.base_url:
            raise SystemExit("--base-url is required for provider=openai_compatible")
        return OpenAICompatibleProvider(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            model_version=args.model_version,
            quantization=args.quantization,
            context_window=args.context_window,
        )
    raise SystemExit(f"Unknown provider: {args.provider}")


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def run_generation_tasks(provider: ILLMProvider, schema: dict) -> list[dict]:
    tasks = load_json(TASKS_DIR / "mcq_generation_tasks.json")
    results = []
    for task in tasks:
        prompt = f"Context:\n{task['context']}\n\nInstruction:\n{task['instruction']}"
        try:
            result = provider.generate_structured(prompt, json_schema=schema, max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            results.append({"task_id": task["id"], "error": str(exc)})
            continue

        structural = validators.grade_structural(result.text, schema)
        row = {
            "task_id": task["id"],
            "domain": task["domain"],
            "difficulty": task["difficulty"],
            "bloom_level": task["bloom_level"],
            "latency_seconds": result.latency_seconds,
            "tokens_per_second": result.tokens_per_second,
            "structural_pass": structural.passed,
            "structural_details": structural.details,
        }
        if structural.passed:
            mcq = structural.details["parsed"]
            row["option_conciseness"] = validators.grade_option_conciseness(mcq["options"]).__dict__
            row["distractor_hygiene"] = validators.grade_distractor_hygiene(mcq["options"]).__dict__
            row["single_correct"] = validators.grade_single_correct(mcq).__dict__
        results.append(row)
    return results


def run_verification_tasks(provider: ILLMProvider, schema: dict) -> list[dict]:
    tasks = load_json(TASKS_DIR / "verification_tasks.json")
    results = []
    for task in tasks:
        options_text = "\n".join(f"{i}: {opt}" for i, opt in enumerate(task["options"]))
        prompt = (
            f"Question: {task['question']}\nOptions:\n{options_text}\n\n"
            f"A generator claims option index {task['claimed_correct_option']} is correct. "
            "Independently derive the correct answer from first principles and decide "
            "whether the claim is PASS (correct), FAIL (incorrect), or UNCERTAIN "
            "(insufficient evidence to decide)."
        )
        try:
            result = provider.generate_structured(prompt, json_schema=schema, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            results.append({"task_id": task["id"], "error": str(exc)})
            continue

        structural = validators.grade_structural(result.text, schema)
        row = {
            "task_id": task["id"],
            "domain": task["domain"],
            "latency_seconds": result.latency_seconds,
            "tokens_per_second": result.tokens_per_second,
            "structural_pass": structural.passed,
        }
        if structural.passed:
            accuracy = validators.grade_verification_accuracy(
                structural.details["parsed"], task["expected_verdict"]
            )
            row["verification_accuracy"] = accuracy.__dict__
        results.append(row)
    return results


def run_json_stress_tasks(provider: ILLMProvider, schema: dict) -> list[dict]:
    tasks = load_json(TASKS_DIR / "json_stress_tasks.json")
    results = []
    for task in tasks:
        try:
            result = provider.generate_structured(task["prompt"], json_schema=schema, max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            results.append({"task_id": task["id"], "error": str(exc)})
            continue
        structural = validators.grade_structural(result.text, schema)
        results.append(
            {
                "task_id": task["id"],
                "description": task["description"],
                "latency_seconds": result.latency_seconds,
                "tokens_per_second": result.tokens_per_second,
                "structural_pass": structural.passed,
                "structural_details": {} if structural.passed else structural.details,
            }
        )
    return results


def run_math_computation_tasks(provider: ILLMProvider) -> list[dict]:
    tasks = load_json(TASKS_DIR / "math_computation_tasks.json")
    answer_schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }
    results = []
    for task in tasks:
        prompt = (
            f"{task['prompt']} Respond with a JSON object "
            '{"answer": "<result, in plain sympy-parseable notation, e.g. 3*x**2 + 1, '
            'or [2, 3] for multiple solutions, or (3, 2) for a coordinate pair>"}.'
        )
        try:
            result = provider.generate_structured(prompt, json_schema=answer_schema, max_tokens=128)
        except Exception as exc:  # noqa: BLE001
            results.append({"task_id": task["id"], "error": str(exc)})
            continue
        grade = validators.grade_math_answer(result.text, task)
        results.append(
            {
                "task_id": task["id"],
                "latency_seconds": result.latency_seconds,
                "tokens_per_second": result.tokens_per_second,
                "correct": grade.passed,
                "details": grade.details,
            }
        )
    return results


def summarize(rows: list[dict], pass_key: str) -> dict:
    total = len(rows)
    errored = sum(1 for r in rows if "error" in r)
    passed = sum(1 for r in rows if r.get(pass_key) is True)
    latencies = [r["latency_seconds"] for r in rows if "latency_seconds" in r]
    tps = [r["tokens_per_second"] for r in rows if r.get("tokens_per_second")]
    return {
        "total": total,
        "errored": errored,
        "pass_rate": passed / total if total else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "median_tokens_per_second": statistics.median(tps) if tps else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["ollama", "llamacpp", "openai_compatible"], required=True)
    parser.add_argument("--model", required=True, help="Model name/tag as known to the provider")
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--quantization", default=None, help="e.g. Q4_K_M")
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the pre-flight provider reachability check (for dry runs against a mock).",
    )
    args = parser.parse_args()

    provider = build_provider(args)

    if not args.skip_health_check and not provider.health_check():
        raise SystemExit(
            f"Provider '{args.provider}' at model '{args.model}' failed health check. "
            "Is the local LLM server running and reachable? "
            "(This benchmark must run on the target 8GB-GPU workstation, not a sandbox "
            "without a GPU/model runtime -- see README.md.)"
        )

    mcq_schema = load_json(SCHEMAS_DIR / "mcq_schema.json")
    verification_schema = load_json(SCHEMAS_DIR / "verification_schema.json")

    monitor = ResourceMonitor(interval_seconds=1.0)
    monitor.start()

    print("Running MCQ generation tasks...")
    generation_results = run_generation_tasks(provider, mcq_schema)
    print("Running answer verification tasks...")
    verification_results = run_verification_tasks(provider, verification_schema)
    print("Running JSON-compliance stress tasks...")
    json_stress_results = run_json_stress_tasks(provider, mcq_schema)
    print("Running deterministic math-computation tasks...")
    math_results = run_math_computation_tasks(provider)

    resource_summary = monitor.stop()

    report = {
        "run_metadata": {
            "provider": provider.metadata().__dict__,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "resource_usage": resource_summary,
        "generation": {
            "summary": summarize(generation_results, "structural_pass"),
            "results": generation_results,
        },
        "verification": {
            "summary": summarize(verification_results, "structural_pass"),
            "verification_accuracy_rate": _accuracy_rate(verification_results),
            "results": verification_results,
        },
        "json_stress": {
            "summary": summarize(json_stress_results, "structural_pass"),
            "results": json_stress_results,
        },
        "math_computation": {
            "summary": summarize(math_results, "correct"),
            "results": math_results,
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone. Full report written to {report_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "run_metadata"}, indent=2)[:2000])


def _accuracy_rate(rows: list[dict]) -> float | None:
    graded = [r["verification_accuracy"]["passed"] for r in rows if "verification_accuracy" in r]
    return sum(graded) / len(graded) if graded else None


if __name__ == "__main__":
    main()
