"""Generation job executor -- Phase 2 scope.

Runs a GenerationJob's plan against a configured ILLMProvider using
manually-supplied per-topic context (no RAG yet -- that's Phase 3). Each
plan cell requests one MCQ at a time (Phase 5 introduces real batching of
10-30 questions per call, per docs/PHASE0-DESIGN.md section 11); structural
validation runs immediately on every response, generation and validation
still deliberately kept as separate function calls/stages even though
Phase 2 doesn't yet have a queue between them.

Every candidate — valid or not — is persisted with full provenance
(job, topic, prompt template, model, model version) before any status
decision is made, per the "never lose provenance" rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from backend.database.models import (
    GenerationJob,
    GenerationJobTopic,
    GenerationModel,
    MCQCandidate,
    MCQOption,
    MCQValidationResult,
    PromptTemplate,
)
from backend.domain.enums import BloomLevel, DifficultyLevel, GenerationJobStatus, MCQStatus
from backend.generation.planner import allocate_topic_counts, build_plan
from backend.validation.structural import validate_mcq_structure
from llm.providers.base import ILLMProvider

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "llm" / "schemas" / "mcq_schema.json"
MCQ_SCHEMA = json.loads(_SCHEMA_PATH.read_text())


def run_generation_job(
    db: Session,
    job: GenerationJob,
    provider: ILLMProvider,
    prompt_template: PromptTemplate,
) -> GenerationJob:
    """Synchronously executes `job` end to end (Phase 2 has no queue/worker
    boundary between "start" and "run" -- see backend/workers/tasks.py for
    where this gets wrapped for async execution).
    """
    job.status = GenerationJobStatus.RUNNING
    db.flush()

    topic_weights = {t.id: t.weight for t in job.topics}
    topic_targets = allocate_topic_counts(job.requested_count, topic_weights)
    for topic in job.topics:
        topic.target_count = topic_targets.get(topic.id, 0)
    db.flush()

    plan = build_plan(topic_targets, job.difficulty_distribution, job.bloom_distribution)

    topics_by_id = {t.id: t for t in job.topics}
    generated = approved_eligible = rejected = 0

    for cell in plan:
        topic = topics_by_id[cell.generation_job_topic_id]
        for _ in range(cell.count):
            candidate, is_valid = _generate_one(db, job, topic, provider, prompt_template, cell)
            generated += 1
            if is_valid:
                approved_eligible += 1
            else:
                rejected += 1

    job.generated_count = generated
    job.rejected_count = rejected
    # Phase 2 has no independent fact/dedup/quality pipeline yet (Phase 7),
    # so "approved_count" here means "reached PENDING_REVIEW", not
    # APPROVED -- an actual human reviewer decision is still required
    # before anything is APPROVED. approved_count on the job row is
    # updated for real once reviews start landing (see api/routers/questions.py).
    job.status = GenerationJobStatus.COMPLETED
    from datetime import datetime, timezone

    job.completed_at = datetime.now(timezone.utc)
    db.flush()
    return job


def _generate_one(
    db: Session,
    job: GenerationJob,
    topic: GenerationJobTopic,
    provider: ILLMProvider,
    prompt_template: PromptTemplate,
    cell,
) -> tuple[MCQCandidate, bool]:
    context = topic.manual_context or ""
    prompt = (
        f"Context:\n{context}\n\n"
        f"Instruction:\nGenerate ONE multiple-choice question about "
        f"'{topic.topic_label_snapshot}' at Bloom level '{cell.bloom_level}' and "
        f"difficulty '{cell.difficulty}', with {job.option_count} options, grounded "
        f"only in the supplied context. If the context does not support a "
        f"confident, factually grounded question, set confidence below 0.3."
    )

    result = provider.generate_structured(
        prompt,
        json_schema=MCQ_SCHEMA,
        temperature=prompt_template.temperature,
        top_p=prompt_template.top_p,
        top_k=prompt_template.top_k,
        repeat_penalty=prompt_template.repeat_penalty,
        max_tokens=prompt_template.max_tokens,
    )

    candidate = MCQCandidate(
        generation_job_id=job.id,
        generation_job_topic_id=topic.id,
        prompt_template_id=prompt_template.id,
        generation_model_id=job.model_id,
        model_version=provider.metadata().model_version,
        question_stem="",
        bloom_level=BloomLevel(cell.bloom_level),
        difficulty=DifficultyLevel(cell.difficulty),
        status=MCQStatus.GENERATED,
    )
    db.add(candidate)
    db.flush()

    parsed = _try_parse_json(result.text)
    if parsed is None:
        candidate.question_stem = "(unparseable model output)"
        candidate.status = MCQStatus.INVALID
        db.add(
            MCQValidationResult(
                mcq_candidate_id=candidate.id,
                validator_name="structural",
                stage="json_parse",
                result="fail",
                details={"raw_text": result.text[:2000]},
            )
        )
        db.flush()
        return candidate, False

    validation = validate_mcq_structure(parsed, MCQ_SCHEMA, max_option_words=6)
    candidate.question_stem = parsed.get("question", "")
    candidate.explanation = parsed.get("explanation")
    candidate.confidence = parsed.get("confidence")

    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="structural",
            stage="structural",
            result="pass" if validation.passed else "fail",
            details={"reasons": validation.reasons, **validation.details},
        )
    )

    if not validation.passed:
        candidate.status = MCQStatus.INVALID
        db.flush()
        return candidate, False

    for idx, option_text in enumerate(parsed["options"]):
        db.add(
            MCQOption(
                mcq_candidate_id=candidate.id,
                option_index=idx,
                text=option_text,
                is_correct=(idx == parsed["correct_option"]),
            )
        )

    # Phase 2 stops here: STRUCTURE_VALIDATED -> straight to PENDING_REVIEW,
    # skipping FACT_VALIDATED / DEDUPLICATED / QUALITY_CHECKED, which don't
    # exist until Phase 7. This is intentional and documented, not a bug --
    # every Phase-2-generated question needs human review before approval
    # precisely because independent fact verification isn't wired up yet.
    candidate.status = MCQStatus.PENDING_REVIEW
    db.flush()
    return candidate, True


def _try_parse_json(text: str) -> dict | None:
    import re

    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def get_or_create_generation_model(
    db: Session, *, name: str, provider_type: str, version: str, quantization: str | None, context_window: int
) -> GenerationModel:
    existing = (
        db.query(GenerationModel)
        .filter_by(name=name, version=version, quantization=quantization)
        .one_or_none()
    )
    if existing:
        return existing
    model = GenerationModel(
        name=name,
        provider_type=provider_type,
        version=version,
        quantization=quantization,
        context_window=context_window,
    )
    db.add(model)
    db.flush()
    return model


def get_or_create_default_prompt_template(db: Session, model_id: int) -> PromptTemplate:
    existing = (
        db.query(PromptTemplate)
        .filter_by(prompt_key="mcq_generation", version=1)
        .one_or_none()
    )
    if existing:
        return existing
    template = PromptTemplate(
        prompt_key="mcq_generation",
        version=1,
        template_text=(
            "Context:\n{context}\n\nInstruction:\nGenerate ONE multiple-choice "
            "question about '{topic}' at Bloom level '{bloom_level}' and "
            "difficulty '{difficulty}', with {option_count} options, grounded "
            "only in the supplied context."
        ),
        model_id=model_id,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=512,
    )
    db.add(template)
    db.flush()
    return template
