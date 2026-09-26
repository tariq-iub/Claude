"""Generation job executor.

Runs a GenerationJob's plan against a configured ILLMProvider. Context for
each topic comes from one of two sources, selected by
`job.source_policy["local_docs"]`:

  - Manual-context mode: `GenerationJobTopic.manual_context` is used verbatim.
  - RAG mode (Phase 3/4): a Topic Knowledge Pack is built by retrieving the
    topic's ingested document/web chunks from the vector store, and the
    resulting citation-tagged text is used instead -- falling back to
    `manual_context` if retrieval finds no evidence at all.

Phase 5 adds real batching (docs/PHASE0-DESIGN.md section 11: 10-30
questions per LLM call, never one call per question and never a whole
job's worth in one call) and an over-generation loop that keeps
requesting more until the job's requested_count of PENDING_REVIEW-eligible
candidates is reached or `job.max_attempts` rounds are exhausted
(section 30). Generation and validation remain separate function
calls/stages even within a batch: every item in a batch response is
structurally validated independently before any status decision.

Every candidate — valid or not — is persisted with full provenance
(job, topic, prompt template, model, model version, and — in RAG mode —
the exact source chunks it was grounded in, via MCQSource) before any
status decision is made, per the "never lose provenance" rule.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import (
    GenerationJob,
    GenerationJobTopic,
    GenerationMetric,
    GenerationModel,
    MCQCandidate,
    MCQOption,
    MCQSource,
    MCQValidationResult,
    PromptTemplate,
)
from backend.domain.enums import BloomLevel, DifficultyLevel, GenerationJobStatus, MCQStatus
from backend.embeddings.base import IEmbeddingProvider
from backend.generation.planner import PlanCell, allocate_topic_counts, build_plan
from backend.validation.structural import validate_mcq_structure
from llm.providers.base import ILLMProvider
from rag.knowledge_pack import build_topic_knowledge_pack
from rag.vectorstore import IVectorStore
from rag.web.injection_defense import wrap_context_as_data

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "llm" / "schemas" / "mcq_schema.json"
MCQ_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

_QUESTION_TYPE_INSTRUCTIONS = {
    "single_best_answer": "Ask a direct question with a single best-answer option.",
    "scenario_based": "Frame the question as a short applied scenario or word problem, then ask for the correct answer.",
    "negative": (
        "Phrase the question so the student must identify the one option that is "
        "FALSE, incorrect, or an exception (e.g. 'Which of the following is NOT...')."
    ),
    "definition_recall": "Ask the student to match a term to its correct defining property or description.",
}


def run_generation_job(
    db: Session,
    job: GenerationJob,
    provider: ILLMProvider,
    prompt_template: PromptTemplate,
    *,
    vector_store: IVectorStore | None = None,
    embedding_provider: IEmbeddingProvider | None = None,
) -> GenerationJob:
    """Synchronously executes `job` end to end (there is no queue/worker
    boundary between "start" and "run" independent of this function --
    see backend/workers/tasks.py for where this gets wrapped for async
    execution).

    `vector_store`/`embedding_provider` are required only when
    `job.source_policy["local_docs"]` is true (RAG mode); manual-context
    jobs never touch them.
    """
    job.status = GenerationJobStatus.RUNNING
    db.flush()

    topics_by_id = {t.id: t for t in job.topics}
    topic_weights = {t.id: t.weight for t in job.topics}

    total_generated = total_rejected = 0
    round_number = 0
    max_rounds = min(job.max_attempts, settings.generation_max_rounds_ceiling)

    # Round 0 over-generates against over_generation_factor (section 30):
    # if the target is 100 approved-eligible questions and history/default
    # suggests ~40% get rejected structurally, asking for exactly 100
    # would under-deliver against the human review queue's real target.
    remaining_target = math.ceil(job.requested_count * job.over_generation_factor)

    while round_number < max_rounds and remaining_target > 0:
        topic_targets = allocate_topic_counts(remaining_target, topic_weights)
        for topic in job.topics:
            topic.target_count = topic_targets.get(topic.id, 0) + (topic.target_count if round_number > 0 else 0)
        db.flush()

        plan = build_plan(
            topic_targets, job.difficulty_distribution, job.bloom_distribution, job.question_type_distribution
        )

        round_generated = round_valid = 0
        for cell in plan:
            topic = topics_by_id[cell.generation_job_topic_id]
            remaining_in_cell = cell.count
            while remaining_in_cell > 0:
                batch_n = min(remaining_in_cell, settings.generation_max_batch_size)
                results = _generate_batch_for_cell(
                    db, job, topic, provider, prompt_template, cell, batch_n,
                    vector_store=vector_store, embedding_provider=embedding_provider,
                )
                round_generated += len(results)
                round_valid += sum(1 for _candidate, is_valid in results if is_valid)
                # Always consume the full requested batch_n slot for this
                # attempt, whether the call fully succeeded, partially
                # succeeded (model legitimately returned fewer items than
                # asked), or failed outright (one INVALID candidate
                # recording the failure). This keeps the inner loop
                # bounded and simple -- true shortfall compensation is the
                # outer round-based over-generation loop's job, not an
                # unbounded retry here.
                remaining_in_cell -= batch_n

        total_generated += round_generated
        total_rejected += round_generated - round_valid
        accept_rate = round_valid / round_generated if round_generated else 0.0

        db.add(GenerationMetric(generation_job_id=job.id, metric_name=f"round_{round_number}_generated", metric_value=round_generated))
        db.add(GenerationMetric(generation_job_id=job.id, metric_name=f"round_{round_number}_valid", metric_value=round_valid))
        db.add(GenerationMetric(generation_job_id=job.id, metric_name=f"round_{round_number}_accept_rate", metric_value=accept_rate))
        db.flush()

        pending_review_count = (
            db.query(MCQCandidate)
            .filter_by(generation_job_id=job.id, status=MCQStatus.PENDING_REVIEW)
            .count()
        )
        shortfall = job.requested_count - pending_review_count
        round_number += 1

        if shortfall <= 0:
            remaining_target = 0
            break

        # Scale the next round's ask by the observed accept rate so far,
        # never trusting a single low-sample round to justify an extreme
        # multiplier (accept_rate is floored at 0.2 -> at most 5x the raw
        # shortfall per round) -- an optimistic first-round assumption
        # (over_generation_factor) already covered round 0.
        effective_rate = max(accept_rate, 0.2) if round_generated > 0 else 1.0
        remaining_target = math.ceil(shortfall / effective_rate)

    final_pending_review_count = (
        db.query(MCQCandidate).filter_by(generation_job_id=job.id, status=MCQStatus.PENDING_REVIEW).count()
    )
    db.add(
        GenerationMetric(
            generation_job_id=job.id,
            metric_name="target_met",
            metric_value=1.0 if final_pending_review_count >= job.requested_count else 0.0,
        )
    )
    db.add(GenerationMetric(generation_job_id=job.id, metric_name="rounds_used", metric_value=round_number))

    job.generated_count = total_generated
    job.rejected_count = total_rejected
    # No independent fact/dedup/quality pipeline yet (Phase 7), so
    # "reaching PENDING_REVIEW" is as far as this job's own counters go --
    # an actual human reviewer decision is still required before anything
    # is APPROVED. job.approved_count is updated for real once reviews
    # start landing (see api/routers/questions.py).
    job.status = GenerationJobStatus.COMPLETED
    job.completed_at = datetime.now(timezone.utc)
    db.flush()
    return job


def resolve_topic_context(
    job: GenerationJob,
    topic: GenerationJobTopic,
    *,
    vector_store: IVectorStore | None,
    embedding_provider: IEmbeddingProvider | None,
) -> tuple[str, list[int]]:
    """Returns (context_text, source_chunk_ids). source_chunk_ids is empty
    for manual-context jobs and for RAG jobs whose fallback to
    manual_context was used because retrieval found no evidence.
    """
    use_rag = bool(job.source_policy.get("local_docs"))
    if use_rag and vector_store is not None and embedding_provider is not None:
        pack = build_topic_knowledge_pack(
            vector_store,
            embedding_provider,
            external_subject_id=job.external_subject_id,
            external_topic_id=topic.external_topic_id,
            topic_label=topic.topic_label_snapshot,
        )
        if pack.has_evidence:
            context = pack.as_context_text()
            if topic.manual_context:
                context = f"{context}\n\n{topic.manual_context}"
            return context, pack.source_chunk_ids

    # Manual-context mode, or RAG mode with no ingested evidence yet.
    return topic.manual_context or "", []


def build_batch_schema(item_schema: dict, max_items: int) -> dict:
    """Wraps a single-item JSON Schema into a batch-response schema:
    {"items": [<item>, ...]}. Built dynamically from the same MCQ_SCHEMA
    used for single-item validation so the two never drift apart.
    """
    return {
        "type": "object",
        "required": ["items"],
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_items,
                "items": item_schema,
            }
        },
    }


def _build_prompt(job: GenerationJob, topic: GenerationJobTopic, cell: PlanCell, context: str, batch_n: int) -> str:
    context_block = wrap_context_as_data(context) if context else "(no supporting context available)"
    type_instruction = _QUESTION_TYPE_INSTRUCTIONS.get(cell.question_type, _QUESTION_TYPE_INSTRUCTIONS["single_best_answer"])
    return (
        f"Context:\n{context_block}\n\n"
        f"Instruction:\nGenerate exactly {batch_n} DISTINCT multiple-choice questions about "
        f"'{topic.topic_label_snapshot}' at Bloom level '{cell.bloom_level}' and "
        f"difficulty '{cell.difficulty}', with {job.option_count} options each, grounded "
        f"only in the supplied context. {type_instruction} Each question must test a "
        f"different fact or aspect of the topic -- do not paraphrase the same question. "
        f"If the context cannot support {batch_n} distinct, factually grounded questions, "
        f"return fewer items rather than fabricating extras or repeating one. "
        f"Return a JSON object with a single \"items\" array of question objects. "
        f"If a question's context is insufficient for a confident, factually grounded "
        f"answer, set that item's confidence below 0.3."
    )


def _generate_batch_for_cell(
    db: Session,
    job: GenerationJob,
    topic: GenerationJobTopic,
    provider: ILLMProvider,
    prompt_template: PromptTemplate,
    cell: PlanCell,
    batch_n: int,
    *,
    vector_store: IVectorStore | None,
    embedding_provider: IEmbeddingProvider | None,
) -> list[tuple[MCQCandidate, bool]]:
    context, source_chunk_ids = resolve_topic_context(
        job, topic, vector_store=vector_store, embedding_provider=embedding_provider
    )
    prompt = _build_prompt(job, topic, cell, context, batch_n)
    batch_schema = build_batch_schema(MCQ_SCHEMA, max(batch_n, 1))

    result = provider.generate_structured(
        prompt,
        json_schema=batch_schema,
        temperature=prompt_template.temperature,
        top_p=prompt_template.top_p,
        top_k=prompt_template.top_k,
        repeat_penalty=prompt_template.repeat_penalty,
        max_tokens=prompt_template.max_tokens * max(1, batch_n // 4 or 1),
    )

    parsed_batch = _try_parse_json(result.text)
    items = parsed_batch.get("items") if isinstance(parsed_batch, dict) else None
    if not isinstance(items, list):
        # The whole batch call failed to produce a usable shape. Persist
        # one INVALID candidate recording the raw failure, rather than
        # silently losing the fact that this cell/batch attempt happened
        # (provenance still applies to failed attempts, per module docstring).
        candidate = _new_candidate(db, job, topic, prompt_template, provider, cell)
        candidate.question_stem = "(unparseable batch response)"
        candidate.status = MCQStatus.INVALID
        db.add(
            MCQValidationResult(
                mcq_candidate_id=candidate.id,
                validator_name="structural",
                stage="batch_json_parse",
                result="fail",
                details={"raw_text": result.text[:2000]},
            )
        )
        db.flush()
        return [(candidate, False)]

    outcomes = []
    for raw_item in items:
        candidate = _new_candidate(db, job, topic, prompt_template, provider, cell)
        is_valid = _apply_parsed_item(db, candidate, raw_item, source_chunk_ids)
        outcomes.append((candidate, is_valid))
    return outcomes


def _new_candidate(
    db: Session,
    job: GenerationJob,
    topic: GenerationJobTopic,
    prompt_template: PromptTemplate,
    provider: ILLMProvider,
    cell: PlanCell,
) -> MCQCandidate:
    candidate = MCQCandidate(
        generation_job_id=job.id,
        generation_job_topic_id=topic.id,
        prompt_template_id=prompt_template.id,
        generation_model_id=job.model_id,
        model_version=provider.metadata().model_version,
        question_stem="",
        bloom_level=BloomLevel(cell.bloom_level),
        difficulty=DifficultyLevel(cell.difficulty),
        question_type=cell.question_type,
        status=MCQStatus.GENERATED,
    )
    db.add(candidate)
    db.flush()
    return candidate


def _apply_parsed_item(db: Session, candidate: MCQCandidate, raw_item, source_chunk_ids: list[int]) -> bool:
    """Validates one already-parsed item dict and writes its options/
    sources/status onto `candidate`. Returns True if it reached
    PENDING_REVIEW."""
    if not isinstance(raw_item, dict):
        candidate.question_stem = "(malformed batch item)"
        candidate.status = MCQStatus.INVALID
        db.add(
            MCQValidationResult(
                mcq_candidate_id=candidate.id,
                validator_name="structural",
                stage="batch_item_shape",
                result="fail",
                details={"reason": "item_is_not_an_object"},
            )
        )
        db.flush()
        return False

    validation = validate_mcq_structure(raw_item, MCQ_SCHEMA, max_option_words=6)
    candidate.question_stem = raw_item.get("question", "")
    candidate.explanation = raw_item.get("explanation")
    candidate.confidence = raw_item.get("confidence")

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
        return False

    for idx, option_text in enumerate(raw_item["options"]):
        db.add(
            MCQOption(
                mcq_candidate_id=candidate.id,
                option_index=idx,
                text=option_text,
                is_correct=(idx == raw_item["correct_option"]),
            )
        )

    for chunk_id in source_chunk_ids:
        db.add(MCQSource(mcq_candidate_id=candidate.id, document_chunk_id=chunk_id))

    # Structural validation only so far -> straight to PENDING_REVIEW,
    # skipping FACT_VALIDATED / DEDUPLICATED / QUALITY_CHECKED, which don't
    # exist until Phase 7. Every question generated so far needs human
    # review before approval precisely because independent fact
    # verification isn't wired up yet.
    candidate.status = MCQStatus.PENDING_REVIEW
    db.flush()
    return True


def _generate_one(
    db: Session,
    job: GenerationJob,
    topic: GenerationJobTopic,
    provider: ILLMProvider,
    prompt_template: PromptTemplate,
    cell,
    *,
    vector_store: IVectorStore | None = None,
    embedding_provider: IEmbeddingProvider | None = None,
) -> tuple[MCQCandidate, bool]:
    """Single-item generation path, used by the question-review
    'regenerate' endpoint (one replacement question at a time) rather
    than the main job executor, which batches (see
    `_generate_batch_for_cell`). Kept separate because a regenerate
    request always wants exactly one item back synchronously, and
    forcing it through the batch-of-N schema machinery for N=1 would add
    complexity with no benefit at that call site.
    """
    context, source_chunk_ids = resolve_topic_context(
        job, topic, vector_store=vector_store, embedding_provider=embedding_provider
    )
    context_block = wrap_context_as_data(context) if context else "(no supporting context available)"
    type_instruction = _QUESTION_TYPE_INSTRUCTIONS.get(cell.question_type, _QUESTION_TYPE_INSTRUCTIONS["single_best_answer"])
    prompt = (
        f"Context:\n{context_block}\n\n"
        f"Instruction:\nGenerate ONE multiple-choice question about "
        f"'{topic.topic_label_snapshot}' at Bloom level '{cell.bloom_level}' and "
        f"difficulty '{cell.difficulty}', with {job.option_count} options, grounded "
        f"only in the supplied context. {type_instruction} If the context does not "
        f"support a confident, factually grounded question, set confidence below 0.3."
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

    candidate = _new_candidate(db, job, topic, prompt_template, provider, cell)

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

    is_valid = _apply_parsed_item(db, candidate, parsed, source_chunk_ids)
    return candidate, is_valid


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
