# Phase 1 Benchmark Report — {{MODEL_NAME}}

Filled in from `results/{{MODEL_NAME}}/report.json` plus expert review.
Do not fill in any field with an estimate — leave it `TBD` if not measured.

## Model identity

| Field | Value |
|---|---|
| Model | |
| Parameters | |
| Quantization | |
| Model file size (disk) | |
| Provider / runtime | Ollama / llama.cpp / OpenAI-compatible |
| Context window used | |

## Automated results (from report.json)

| Metric | Value |
|---|---|
| MCQ generation — structural (JSON schema) pass rate | |
| MCQ generation — median latency (s) | |
| MCQ generation — median tokens/sec | |
| Option conciseness pass rate (≤6 words) | |
| Distractor hygiene pass rate (no exact duplicates) | |
| JSON-stress suite pass rate | |
| Math computation — SymPy-verified correctness rate | |
| Answer verification — verdict accuracy | |
| Answer verification — abstention (UNCERTAIN) rate | |

## Resource usage (from report.json → resource_usage)

| Metric | Value |
|---|---|
| GPU monitoring available (`nvidia-smi` found) | |
| Peak GPU memory used (MB) | |
| Average GPU utilization (%) | |
| Peak system RAM used (MB) | |

## Expert-scored quality (manual — see README.md step 4)

Score each on 1-5. Reviewer(s): _______________

| Dimension | Score (1-5) | Notes |
|---|---|---|
| Factual correctness | | |
| Topic/syllabus relevance | | |
| Question clarity | | |
| Distractor plausibility (not just structural hygiene) | | |
| Difficulty-label accuracy | | |
| Bloom-level-label accuracy | | |
| Math/physics/chemistry notation correctness (rendered, not just parseable) | | |

## Overall recommendation

- [ ] Candidate generation model
- [ ] Candidate verifier model
- [ ] Rejected — reason: _______________

## Notes / anomalies observed during the run

(e.g. thermal throttling, OOM, unusually high rejection on a specific
domain, JSON parse failures that repeated across the same task type)
