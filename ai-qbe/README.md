# AI Academic Question Bank Engine (AI-QBE)

Self-hosted, privacy-preserving pipeline that turns a university's syllabus
into large, validated, syllabus-aligned MCQ question banks using a locally
hosted LLM, controlled RAG retrieval, deterministic validation, and human
academic review.

This directory currently contains the **Phase 0** deliverable only:
requirements, architecture, and design decisions. No application code has
been written yet — per the phased plan, Phase 1 (local LLM benchmark) starts
only after Phase 0 is reviewed and approved.

## Contents

- [`docs/PHASE0-DESIGN.md`](docs/PHASE0-DESIGN.md) — the full Phase 0 technical
  design: architecture, data flow, ERD, model/embedding/vector-store choices,
  RAG and generation architecture, validation architecture, math/physics/
  chemistry strategy, security architecture, REST API surface, project
  structure, deployment architecture, performance constraints, risks, roadmap,
  and phase-by-phase acceptance criteria.
- [`docs/schema.sql`](docs/schema.sql) — PostgreSQL DDL for the AI-QBE metadata
  schema described in the ERD (generation jobs, sources, chunks, candidates,
  validation results, reviews, versions — provenance-complete).

## Status

**Phase 0 — Requirements & Architecture.** Awaiting review/approval before
Phase 1 (local LLM benchmarking on the 8GB-VRAM target workstation) begins.
No numbers in the design doc (throughput, accuracy, VRAM use) are claimed as
measured — they are estimates to be replaced with real benchmark data in
Phase 1, and the doc calls this out explicitly wherever it applies.
