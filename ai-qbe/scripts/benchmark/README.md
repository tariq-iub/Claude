# Phase 1 — Local LLM Benchmark Harness

Benchmarks candidate quantized local models on the target 8GB-VRAM / 32GB-RAM
workstation, per `docs/PHASE0-DESIGN.md` sections 5 and 20 (Phase 1).

## Why this must run on the real workstation

This harness was developed and unit-tested in a sandbox with **no GPU, no
Ollama/llama.cpp runtime, and no network access to model registries** — so
no benchmark numbers exist yet, and none are claimed anywhere in this repo.
Running it is the next concrete step, on the actual target machine.

## 1. Install a local runtime and pull candidate models

**Option A — Ollama (recommended default):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M   # verifier-role candidate
```

Confirm GPU offload is active: `ollama ps` should show a nonzero GPU
percentage while a model is loaded; if it falls back to 100% CPU, check
`OLLAMA_NUM_GPU` / your CUDA/ROCm driver install.

**Option B — llama.cpp server**, if you need grammar-constrained decoding
or finer `--n-gpu-layers` control:

```bash
./llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    --n-gpu-layers 999 --ctx-size 4096 --port 8080
```

## 2. Install harness dependencies

```bash
cd ai-qbe
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/benchmark/requirements.txt
```

## 3. Run the benchmark against each candidate model

```bash
python scripts/benchmark/run_benchmark.py \
    --provider ollama \
    --model qwen2.5:7b-instruct-q4_K_M \
    --model-version qwen2.5-7b-instruct \
    --quantization Q4_K_M \
    --context-window 4096 \
    --output-dir ./results/qwen2.5-7b-q4
```

Repeat for every candidate model in the Phase 0 shortlist
(`docs/PHASE0-DESIGN.md` section 5). Run each candidate on an otherwise-idle
GPU/machine and back-to-back with the others so resource numbers are
comparable.

Each run measures automatically:

- **JSON structural compliance** — % of MCQ-generation, verification, and
  JSON-stress tasks whose output parses as JSON and validates against the
  schema in `llm/schemas/`.
- **Option conciseness / distractor hygiene** — % of generated MCQs whose
  options are ≤6 words and contain no exact duplicates (cheap structural
  checks; not the full Phase 7 distractor validator).
- **Deterministic math correctness** — SymPy-verified accuracy on the
  `math_computation_tasks` set (algebra/calculus, independent of the
  model's own claimed confidence).
- **Verification accuracy** — how often the model's independent
  PASS/FAIL/UNCERTAIN verdict on `verification_tasks` matches ground truth
  (half the set has a deliberately wrong claimed answer, to check the
  model actually catches errors rather than rubber-stamping).
- **Latency / tokens-per-second** per call.
- **GPU utilization / VRAM / RAM** sampled once per second for the run
  duration (`nvidia-smi` + `psutil`; if `nvidia-smi` isn't found the report
  says so explicitly rather than omitting the field silently).

## 4. Score academic/science quality (not automatable)

The harness cannot judge question clarity, distractor plausibility, or
factual/scientific correctness of open-ended generations by itself — using
an LLM (or this script) as sole judge of its own output is exactly what the
master prompt forbids. For that dimension:

1. Open `results/<model>/report.json`, pull out the generated MCQs from
   `generation.results[*].structural_details.parsed`.
2. Have a subject-matter reviewer score a sample (recommend: all 40
   generation-task outputs per model, since that's already a manageable
   set) using the rubric in `report_template.md`.
3. Fill the "Expert-Scored Quality" section of `report_template.md` per
   model from that review.

## 5. Compare and select

Copy `report_template.md` per model into `results/<model>/SUMMARY.md`, fill
in the automated numbers from `report.json` plus the expert quality scores,
then compare across candidates side by side before selecting the Phase 2+
default generation model (and, separately, a verifier model — see Phase 0
section 5's generator/verifier split rationale).

## Files

- `run_benchmark.py` — the harness entry point (see `--help`).
- `validators.py` — automated grading (JSON Schema, option/distractor
  hygiene, SymPy math checking, verification-accuracy scoring).
- `resource_monitor.py` — background GPU/RAM sampler.
- `tasks/*.json` — the 100-task benchmark set (40 MCQ-generation, 30
  answer-verification, 20 JSON-compliance-stress, 10 SymPy-checkable math).
- `report_template.md` — the human-fillable comparison report format.
