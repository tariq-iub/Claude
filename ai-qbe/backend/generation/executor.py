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
validated independently before any status decision.

Phase 7 adds the full QA pipeline to that per-item validation, run in
order after structural + notation validation both pass: semantic
deduplication (backend/validation/dedup.py) against every still-alive
candidate ever generated for the subject/topic, distractor quality
(backend/validation/distractor.py), independent answer verification
(backend/validation/answer_verification.py -- SymPy where a clean
computation is extractable, an LLM verifier pass otherwise; never the
generator's own claimed answer taken on faith), a difficulty/Bloom
cross-check (recorded only, never blocking), and a composite quality
score (backend/validation/quality_score.py) derived solely from those
validators' outputs. A candidate can now end at PENDING_REVIEW, INVALID,
REJECTED, DUPLICATE, or LOW_CONFIDENCE (quality score below
`settings.quality_score_low_confidence_threshold`) -- only PENDING_REVIEW
counts toward a job's requested_count target.

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
    MCQQualityScore,
    MCQSource,
    MCQValidationResult,
    PromptTemplate,
)
from backend.domain.enums import BloomLevel, DifficultyLevel, GenerationJobStatus, MCQStatus, TERMINAL_REJECTED_STATUSES
from backend.embeddings.base import IEmbeddingProvider
from backend.generation.planner import PlanCell, allocate_topic_counts, build_plan
from backend.validation.answer_verification import verify_answer
from backend.validation.dedup import ExistingCandidate, check_duplicate
from backend.validation.difficulty_bloom import check_difficulty_bloom
from backend.validation.distractor import validate_distractors
from backend.validation.notation import validate_notation
from backend.validation.quality_score import compute_quality_score
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
    # rejected_count is the broad "did not reach PENDING_REVIEW" bucket
    # (INVALID + REJECTED + DUPLICATE + LOW_CONFIDENCE combined);
    # duplicate_count below is the precise subset of those specifically
    # caught by deduplication, since the schema gives it its own column.
    job.rejected_count = total_rejected
    job.duplicate_count = (
        db.query(MCQCandidate).filter_by(generation_job_id=job.id, status=MCQStatus.DUPLICATE).count()
    )
    # The QA pipeline (Phase 7: dedup, distractor validation, independent
    # answer verification, quality scoring) still stops at PENDING_REVIEW,
    # never APPROVED -- an actual human reviewer decision is still required.
    # job.approved_count is updated for real once reviews start landing
    # (see api/routers/questions.py).
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
        is_valid = _apply_parsed_item(
            db, job, topic, provider, candidate, raw_item, source_chunk_ids,
            context=context, embedding_provider=embedding_provider,
        )
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


def _fetch_dedup_pool(db: Session, job: GenerationJob, topic: GenerationJobTopic, exclude_candidate_id: int) -> list[ExistingCandidate]:
    """Candidates to compare a new one against for semantic deduplication:
    every still-alive (non-terminal-rejected) candidate ever generated for
    the same subject+topic, across ALL jobs -- a growing question bank
    should never accumulate near-duplicates just because they came from
    different generation runs.
    """
    rows = (
        db.query(MCQCandidate)
        .join(GenerationJobTopic, MCQCandidate.generation_job_topic_id == GenerationJobTopic.id)
        .join(GenerationJob, MCQCandidate.generation_job_id == GenerationJob.id)
        .filter(GenerationJob.external_subject_id == job.external_subject_id)
        .filter(GenerationJobTopic.external_topic_id == topic.external_topic_id)
        .filter(MCQCandidate.status.notin_(TERMINAL_REJECTED_STATUSES))
        .filter(MCQCandidate.id != exclude_candidate_id)
        .filter(MCQCandidate.question_stem != "")
        .all()
    )
    return [ExistingCandidate(id=r.id, question_stem=r.question_stem) for r in rows]


def _apply_parsed_item(
    db: Session,
    job: GenerationJob,
    topic: GenerationJobTopic,
    provider: ILLMProvider,
    candidate: MCQCandidate,
    raw_item,
    source_chunk_ids: list[int],
    *,
    context: str = "",
    embedding_provider: IEmbeddingProvider | None = None,
) -> bool:
    """Validates one already-parsed item dict, runs it through the full
    Phase 7 QA pipeline (structural -> notation -> deduplication ->
    distractor quality -> independent answer verification -> difficulty/
    Bloom cross-check -> composite quality score), and writes the result
    onto `candidate`. Returns True if it reached PENDING_REVIEW.
    """
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

    notation = validate_notation(raw_item)
    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="notation",
            stage="notation",
            result=notation.status_label,
            details={"reasons": notation.reasons, "unit_warnings": notation.unit_warnings, **notation.details},
        )
    )
    if not notation.passed:
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
    db.flush()

    options = raw_item["options"]
    correct_option = raw_item["correct_option"]

    # --- Semantic deduplication (4 levels: hash -> lexical -> embedding
    # -> LLM-judge tie-break for the borderline band) ---------------------
    dedup_pool = _fetch_dedup_pool(db, job, topic, exclude_candidate_id=candidate.id)
    dedup = check_duplicate(
        candidate.question_stem, dedup_pool, embedding_provider=embedding_provider, llm_provider=provider
    )
    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="dedup",
            stage="dedup",
            result="fail" if dedup.is_duplicate else "pass",
            details={"method": dedup.method, "matched_candidate_id": dedup.matched_candidate_id, "score": dedup.score, "reason": dedup.reason},
        )
    )
    if dedup.is_duplicate:
        candidate.status = MCQStatus.DUPLICATE
        db.flush()
        return False

    # --- Distractor quality (near-duplicate options, possible second
    # correct answer, length outliers) ------------------------------------
    distractor = validate_distractors(options, correct_option, embedding_provider=embedding_provider)
    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="distractor",
            stage="distractor",
            result="pass" if distractor.passed else "fail",
            details={"reasons": distractor.reasons, "flags": distractor.flags, "embedding_checked": distractor.embedding_checked},
        )
    )
    if not distractor.passed:
        candidate.status = MCQStatus.REJECTED
        db.flush()
        return False

    # --- Independent answer verification (never trust the generator's own
    # claimed correct_option) --------------------------------------------
    answer_verification = verify_answer(
        provider, question_stem=candidate.question_stem, options=options, correct_option=correct_option, context=context
    )
    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="answer_verifier",
            stage="answer_verification",
            result=answer_verification.verdict.lower() if answer_verification.verdict != "UNCERTAIN" else "uncertain",
            details={
                "method": answer_verification.method,
                "confidence": answer_verification.confidence,
                "reason": answer_verification.reason,
                "derived_correct_option": answer_verification.derived_correct_option,
            },
        )
    )
    if answer_verification.verdict == "FAIL":
        candidate.status = MCQStatus.REJECTED
        db.flush()
        return False

    # --- Difficulty/Bloom cross-check: recorded only, never blocking; a
    # reviewer already has change_difficulty/change_bloom actions ---------
    diff_bloom = check_difficulty_bloom(candidate.question_stem, options, candidate.bloom_level, candidate.difficulty)
    db.add(
        MCQValidationResult(
            mcq_candidate_id=candidate.id,
            validator_name="difficulty_bloom",
            stage="difficulty_bloom",
            result="pass" if not diff_bloom.notes else "uncertain",
            details={
                "estimated_bloom": diff_bloom.estimated_bloom.value if diff_bloom.estimated_bloom else None,
                "estimated_difficulty": diff_bloom.estimated_difficulty.value if diff_bloom.estimated_difficulty else None,
                "notes": diff_bloom.notes,
            },
        )
    )

    # --- Composite quality score, derived only from the validators above,
    # never from the generator's own self-reported confidence -------------
    quality = compute_quality_score(
        structural=validation,
        notation=notation,
        answer_verification=answer_verification,
        distractor=distractor,
        dedup=dedup,
        difficulty_bloom=diff_bloom,
        has_sources=len(source_chunk_ids) > 0,
        used_manual_context=bool(topic.manual_context),
        had_any_context=bool(context),
    )
    db.add(
        MCQQualityScore(
            mcq_candidate_id=candidate.id,
            factual_correctness=quality.factual_correctness,
            source_grounding=quality.source_grounding,
            clarity=quality.clarity,
            distractor_quality=quality.distractor_quality,
            single_correctness=quality.single_correctness,
            difficulty_match=quality.difficulty_match,
            bloom_match=quality.bloom_match,
            option_conciseness=quality.option_conciseness,
            notation_validity=quality.notation_validity,
            duplicate_risk=quality.duplicate_risk,
            composite_score=quality.composite_score,
        )
    )

    # A candidate scoring below the configured threshold still isn't
    # discarded (its content may well be salvageable) but is kept out of
    # the PENDING_REVIEW pool that counts toward the job's requested
    # target -- low-confidence items get their own reviewable status
    # rather than diluting the main review queue silently.
    if quality.composite_score < settings.quality_score_low_confidence_threshold:
        candidate.status = MCQStatus.LOW_CONFIDENCE
        db.flush()
        return False

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

    is_valid = _apply_parsed_item(
        db, job, topic, provider, candidate, parsed, source_chunk_ids,
        context=context, embedding_provider=embedding_provider,
    )
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
