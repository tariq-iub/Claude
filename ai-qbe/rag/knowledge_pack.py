"""Topic Knowledge Pack builder (docs/PHASE0-DESIGN.md section 8 / master
prompt section 8).

Phase 3 scope: retrieval-backed knowledge packs assembled from
`document_chunks` via semantic + metadata-filtered search. The structured
fields the master prompt's example shows (definitions, formulas,
relationships, laws, common_misconceptions extracted *from* the chunks,
not just the raw chunk text) require an LLM extraction pass over the
retrieved evidence -- deferred to whichever later phase first needs that
structure (the generation planner, Phase 5), since building it
speculatively now, before any generation prompt actually consumes it,
risks the wrong shape. What Phase 3 delivers instead is exactly what
Phase 2's `manual_context` string slot needs: a bounded, citation-
preserving block of retrieved text the generation executor can drop into
the same prompt position manual context used to occupy.
"""

from __future__ import annotations

import dataclasses

from backend.embeddings.base import IEmbeddingProvider
from rag.vectorstore import IVectorStore, ScoredChunk


@dataclasses.dataclass
class TopicKnowledgePack:
    external_subject_id: str
    external_topic_id: str
    topic_label: str
    retrieved_chunks: list[ScoredChunk]

    # Deferred to Phase 5 (see module docstring): an LLM extraction pass
    # over retrieved_chunks would populate these. Left explicit and empty
    # rather than omitted, so a caller can't mistake "not built yet" for
    # "genuinely has no formulas".
    definitions: list[str] = dataclasses.field(default_factory=list)
    formulas: list[str] = dataclasses.field(default_factory=list)
    common_misconceptions: list[str] = dataclasses.field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return len(self.retrieved_chunks) > 0

    @property
    def source_chunk_ids(self) -> list[int]:
        return [c.document_chunk_id for c in self.retrieved_chunks]

    def as_context_text(self) -> str:
        """Renders retrieved chunks as a citation-tagged context block for
        the generation prompt -- each chunk is tagged with a stable
        [chunk:<id>] marker so a later stage (or a human reviewer) can
        trace a generated question's claim back to the exact chunk it
        came from, per the "never fabricate references" rule.
        """
        if not self.retrieved_chunks:
            return ""
        blocks = []
        for chunk in self.retrieved_chunks:
            location = []
            if chunk.section:
                location.append(f"section: {chunk.section}")
            if chunk.page is not None:
                location.append(f"page: {chunk.page}")
            location_str = f" ({', '.join(location)})" if location else ""
            blocks.append(f"[chunk:{chunk.document_chunk_id}]{location_str}\n{chunk.text}")
        return "\n\n".join(blocks)


def retrieve_chunks(
    vector_store: IVectorStore,
    embedding_provider: IEmbeddingProvider,
    query_text: str,
    *,
    external_subject_id: str,
    external_topic_id: str | None = None,
    top_k: int = 8,
) -> list[ScoredChunk]:
    query_vector = embedding_provider.embed_query(query_text)
    return vector_store.query(
        query_vector,
        external_subject_id=external_subject_id,
        external_topic_id=external_topic_id,
        top_k=top_k,
    )


def build_topic_knowledge_pack(
    vector_store: IVectorStore,
    embedding_provider: IEmbeddingProvider,
    *,
    external_subject_id: str,
    external_topic_id: str,
    topic_label: str,
    query_text: str | None = None,
    top_k: int = 8,
) -> TopicKnowledgePack:
    chunks = retrieve_chunks(
        vector_store,
        embedding_provider,
        query_text or topic_label,
        external_subject_id=external_subject_id,
        external_topic_id=external_topic_id,
        top_k=top_k,
    )
    return TopicKnowledgePack(
        external_subject_id=external_subject_id,
        external_topic_id=external_topic_id,
        topic_label=topic_label,
        retrieved_chunks=chunks,
    )
