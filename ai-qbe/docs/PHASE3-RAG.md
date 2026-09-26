# AI-QBE — Phase 3: Document RAG

Status: **Implemented and tested (91/91 tests passing).** Internet
retrieval (Phase 4) and the fully structured Topic Knowledge Pack fields
(definitions/formulas/misconceptions extracted via an LLM pass, Phase 5)
are still out of scope, by design — see limitations.

## 1. What was implemented

- **`backend/embeddings/`** — the `IEmbeddingProvider` interface, mirrored
  on `ILLMProvider` (Phase 1), with two implementations:
  - `SentenceTransformersEmbeddingProvider` — the real production choice
    (BAAI/bge-small-en-v1.5, per `docs/PHASE0-DESIGN.md` section 6). **Not
    exercised in this sandbox**: downloading model weights from
    huggingface.co is blocked by this environment's network policy
    (confirmed directly — a request to huggingface.co returns a
    proxy-level 403), the same constraint that blocked Phase 1's model
    benchmarking.
  - `HashingEmbeddingProvider` — a dependency-free, deterministic hashed
    bag-of-words vector used as the dev/test default. It captures
    lexical-overlap similarity, not learned semantic similarity, and is
    explicitly documented as never a substitute for a benchmarked
    embedding model — it exists purely so the chunking → embed → vector
    store → retrieval pipeline's plumbing and metadata filtering could
    actually be tested end-to-end here.
- **`rag/vectorstore.py`** — the `IVectorStore` interface and
  `QdrantVectorStore`. This is the one RAG component genuinely exercised
  against real (not mocked) infrastructure in this sandbox: `qdrant-client`
  supports an embedded `location=":memory:"` mode with no server process,
  which behaves identically to a real Qdrant deployment through the same
  API — so `tests/test_vectorstore.py` runs real upserts, cosine-similarity
  queries, metadata-filtered retrieval, point-id-based overwrite
  deduplication, and source-scoped deletion against actual Qdrant code.
- **`rag/extraction.py`** — PDF (via `pypdf`) and plain text/Markdown
  extraction, each returning per-page text (page numbers preserved for
  citation). Verified against a real generated PDF, not a stub.
- **`rag/chunking.py`** — clean → normalize → semantic chunking:
  paragraph/heading-boundary aware packing up to a configurable word
  budget (~400 words ≈ 500 tokens), with heading-line detection for
  `section` labels, oversized-paragraph splitting by sentence, and
  undersized trailing-chunk merging.
- **`rag/ingestion.py`** — ties extraction → chunking → embedding →
  vector-store upsert → `DocumentChunk` persistence together, with
  content-hash deduplication at the `AcademicSource` level (re-uploading
  identical bytes is a no-op, not a duplicate ingestion).
- **`rag/knowledge_pack.py`** — `build_topic_knowledge_pack()`: subject/
  topic-scoped retrieval producing a `TopicKnowledgePack` whose
  `as_context_text()` renders retrieved chunks as a citation-tagged block
  (`[chunk:<id>] (section: ..., page: ...)`) ready to drop into a
  generation prompt.
- **`backend/generation/executor.py` updated** — `resolve_topic_context()`
  now branches on `job.source_policy["local_docs"]`: RAG mode retrieves a
  knowledge pack and uses it (merging in `manual_context` too, if also
  supplied) when evidence exists, falling back to `manual_context` only
  when retrieval finds nothing — a RAG-enabled topic with no ingested
  documents still runs, rather than crashing or silently producing nothing.
  Every RAG-grounded candidate gets `MCQSource` rows recording exactly
  which chunks it was grounded in.
- **`backend/api/routers/sources.py`** — `POST /api/sources` (multipart
  upload: subject/topic/title/author + file, `.pdf`/`.txt`/`.md` only, 50MB
  cap) and `GET /api/sources/{id}` (metadata + chunk count + ingestion
  status).
- **`GenerationJobCreate.use_rag`** — job creation now takes a `use_rag`
  flag; when true, per-topic `manual_context` becomes optional (merged in
  as extra context if supplied) instead of required.
- **`MCQCandidateOut.source_chunk_ids`** — exposed on the question API so
  a reviewer (or an automated citation audit) can see exactly which
  ingested chunks a RAG-grounded question came from.
- **Tests** (`test_chunking.py`, `test_embeddings.py`, `test_vectorstore.py`,
  `test_extraction` cases in `test_ingestion.py`, `test_ingestion.py`,
  `test_knowledge_pack.py`, `test_executor_rag_mode.py`,
  `test_api_rag_flow.py`) — 37 new tests, all passing, on top of Phase 1/2's
  54, for 91 total. Coverage includes: chunk-boundary correctness, embedding
  determinism and normalization, real Qdrant upsert/query/filter/dedup/
  delete behavior, real PDF text extraction, ingestion hash-deduplication,
  topic-scoped retrieval isolation (confirming a Vectors-topic chunk never
  leaks into a Newton's-Laws query), the executor's RAG-vs-manual-context
  branching and citation recording, and a full API flow (upload → dedup
  re-upload → create RAG job → start → verify citations on the resulting
  questions).

```
$ python3 -m pytest tests/ -q
...........................................................................
.....................
91 passed in 9.32s
```

## 2. Design decisions

- **`IEmbeddingProvider` and `IVectorStore` mirror the `ILLMProvider`
  pattern from Phase 1** on purpose: the same "never lock the app to one
  vendor" argument applies to the embedding model and the vector database,
  per `docs/PHASE0-DESIGN.md` sections 6–7.
- **Qdrant's embedded mode, not a hand-rolled in-memory store, backs the
  dev/test default.** Writing a second `IVectorStore` implementation just
  for tests would mean the tested code path and the production code path
  diverge — the whole point of testing `QdrantVectorStore` directly against
  `:memory:` is that the same class, same query logic, same payload
  filtering runs in both dev and production.
- **The hashing embedding provider is lexical, not semantic, and is
  labeled as such everywhere it appears** (docstring, config comment,
  README). This matters because a lexical fallback can *look* like it's
  working in tests — topic-scoped retrieval correctly isolates
  Newton's-Laws from Vectors chunks — while giving materially worse
  results than a real semantic model on paraphrased queries. The topic-
  scoping test deliberately checks metadata filtering, not semantic
  recall, precisely to avoid overclaiming what the hashing provider proves.
- **RAG falls back to `manual_context`, never to silence or a crash.**
  Per Phase 0's "if sufficient evidence is unavailable, do not generate
  the question" instruction (section 20) — but Phase 3 doesn't yet have
  the LLM told "no evidence, don't answer" wired in as a hard gate (that's
  naturally where Phase 7's grounding validator lands); for now, an empty
  context still reaches the generation prompt, which already instructs the
  model to lower its confidence when context is insufficient (Phase 2's
  existing prompt wording). A job for a topic with literally nothing
  ingested still completes rather than failing the whole job.
- **Citations are chunk-id references (`MCQSource` rows +
  `[chunk:<id>]` markers), never inlined source text.** This keeps the
  prompt's evidence traceable back to an exact `DocumentChunk` row (which
  itself points to page/section) without duplicating source content into
  the candidate's own stored fields.
- **The fully structured `TopicKnowledgePack` fields (definitions,
  formulas, common_misconceptions) are deliberately left empty**, not
  half-implemented. Populating them requires an LLM extraction pass over
  retrieved evidence; building that now, before Phase 5's generation
  planner has decided what shape it actually needs, risks the wrong
  interface. The dataclass fields exist (not omitted) so this is visibly a
  "not yet" rather than an unstated gap.

## 3. Configuration

New environment variables (all prefixed `AIQBE_`, see `backend/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `AIQBE_EMBEDDING_PROVIDER_TYPE` | `hashing` | Set to `sentence_transformers` once Phase 1 hardware benchmarking also validates an embedding model choice on real hardware. |
| `AIQBE_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Passed to `SentenceTransformersEmbeddingProvider` when selected. |
| `AIQBE_VECTOR_STORE_LOCATION` | `:memory:` | A real deployment points this at `http://<host>:6333`. |

Uploading a document:

```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Authorization: Bearer $TOKEN" \
  -F "external_subject_id=phys-1" \
  -F "external_topic_id=newtons-laws" \
  -F "title=Lecture Notes Week 3" \
  -F "file=@newtons_laws.pdf"
```

Creating a RAG-enabled generation job: set `"use_rag": true` and omit
`manual_context` on each topic (see `tests/test_api_rag_flow.py` for a
complete request body).

## 4. Limitations

- **No embedding model has actually been benchmarked or run** — same
  network constraint as Phase 1's LLM benchmark. `HashingEmbeddingProvider`
  is a placeholder for pipeline testing, not a retrieval-quality baseline;
  do not infer anything about real semantic retrieval quality from these
  tests passing.
- **No Internet retrieval** — Phase 4 scope. `rag/ingestion.py` is written
  so Phase 4's fetch-and-sanitize step can feed the same
  `ingest_document()` pipeline; only the byte source changes.
- **No LLM-based knowledge-pack structuring** (definitions/formulas/
  misconceptions extracted from evidence) — see design decisions above.
- **No cross-page header/footer stripping** in `clean_text()` — it strips
  standalone page-number lines but can't detect a repeated running header
  without comparing across pages, which the current per-page extraction
  API doesn't expose. Flagged in the function's own docstring rather than
  silently under-cleaning.
- **RAG mode has no hard "insufficient evidence → refuse to generate"
  gate yet.** An empty-context fallback still reaches the LLM prompt
  (which is instructed to lower confidence, not to refuse) — a true gate
  belongs to Phase 7's source-grounding validator, which can inspect
  `MCQSource` (or its absence) on each candidate and reject/flag
  accordingly.
- **`POST /api/sources` accepts `.pdf`/`.txt`/`.md` only** — other formats
  used by real courseware (`.pptx`, `.docx`) are not implemented; adding
  them is a new `rag/extraction.py` function plus a MIME-type entry, not
  an architectural change.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "Upload → clean → chunk → embed → Qdrant → topic knowledge pack
> retrieval demonstrated against real instructor-provided material, with
> citations traceable to source chunk/page."

**Met**, within the limitations above: demonstrated against a real
generated PDF and plain-text upload, through real Qdrant (embedded mode),
with `MCQSource` rows and `[chunk:<id>]` markers making every RAG-grounded
question's evidence traceable to an exact chunk (and, via that chunk, its
page/section). The embedding model behind retrieval is a lexical stand-in,
not the benchmarked semantic model Phase 0 specifies — that swap happens
once Phase 1-style benchmarking is possible on real hardware.
