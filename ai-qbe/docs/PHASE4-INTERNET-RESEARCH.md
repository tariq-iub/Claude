# AI-QBE — Phase 4: Controlled Internet Research

Status: **Implemented and tested (128/128 tests passing).** Live fetches
against the real Internet have not been exercised — this development
sandbox's outbound network policy blocks arbitrary HTTP requests entirely
(confirmed directly: a plain request to `example.com` is rejected by the
proxy with a 403), the same constraint that blocked Phase 1's LLM
benchmark and Phase 3's embedding model. Every component is written
against injectable interfaces and tested for real logic (domain policy,
sanitization, injection defense, chunking/storage) using a `StaticFetcher`
standing in for the network call itself.

## 1. What was implemented

- **`rag/web/domain_policy.py`** — `DomainPolicy`, reading
  `ApprovedDomain`/`BlockedDomain` from the database (both tables already
  existed in the Phase 2 schema, unused until now). **Deny-by-default**:
  a domain that is neither approved nor blocked is rejected — there is no
  code path that returns "allowed" without an `ApprovedDomain` row
  existing for that exact host. Blocked always overrides approved.
  Subdomain matching is exact-host only (`evil.university.edu.attacker.com`
  is not implicitly covered by an approval for `university.edu`).
- **`rag/web/fetch.py`** — `IWebFetcher` interface, `RequestsFetcher` (the
  real implementation: timeouts, a `max_bytes` cap enforced while
  streaming rather than after full download, a content-type allowlist —
  `text/html` and `application/pdf` only — and best-effort `robots.txt`
  compliance that fails open only on a robots.txt *fetch* failure, never
  on a disallow rule), and `StaticFetcher` (canned responses, used
  throughout this sandbox's tests and available for any deployment's own
  deterministic tests).
- **`rag/web/sanitize.py`** — `sanitize_html()`: strips
  `script`/`style`/`nav`/`header`/`footer`/`form`/`iframe`/`aside` tags
  and HTML comments (a documented hiding place for embedded instructions,
  master prompt section 35) before any text extraction, via
  BeautifulSoup.
- **`rag/web/injection_defense.py`** — two independent layers, per
  `docs/PHASE0-DESIGN.md` section 9's "structural, not just prompt
  wording" requirement:
  1. `scrub_untrusted_text()` — a line-level regex pre-filter applied at
     ingestion time (before a chunk is ever embedded or stored), catching
     phrasings like "ignore previous instructions", fake
     `system:`/`assistant:` role markers, ChatML-style `<|im_start|>`
     tokens, and "you are now ..." role-switch attempts. Only the
     matching line is redacted, not the whole page.
  2. `wrap_context_as_data()` — applied at prompt-assembly time (now used
     by `backend/generation/executor.py` for **every** generation prompt,
     not just RAG/web ones), wrapping context in a per-request random
     boundary token with an explicit "this is data, not instructions"
     instruction — so a novel injection pattern that slips past the
     regex layer still faces a second, structurally independent defense.
- **`rag/web/search.py`** — `ISearchAdapter` interface, `SearXNGSearchAdapter`
  (self-hosted search backend, the Phase 0 default choice — avoids a
  commercial search API dependency and keeps query traffic internal), and
  `NullSearchAdapter` (the safe default when no search backend is
  configured: **zero** candidate URLs, never fabricated ones, matching
  "never fabricate references").
- **`rag/web/ingestion.py`** — `ingest_from_url()`: domain-policy check →
  fetch → sanitize (HTML) or extract (PDF) → injection-scrub → the exact
  same `ingest_pages()` chunk/embed/store helper Phase 3's upload path
  uses (extracted via a small refactor of `rag/ingestion.py` so upload and
  URL ingestion share one code path once bytes become `ExtractedPage`s,
  rather than two ingestion implementations that could drift).
- **API additions**:
  - `POST /api/sources/from-url` — fetches and ingests one URL; **403s
    immediately** if the domain isn't approved, before any fetch happens.
  - `POST /api/sources/search-preview` — runs the configured search
    adapter and annotates each result with its domain's current approval
    status, for an administrator to review before deciding what to
    approve. **Never ingests anything itself.**
  - `GET/POST/DELETE /api/domains/approved` and `/api/domains/blocked` —
    Administrator-only domain-list management (`Role.ADMINISTRATOR`), with
    audit events on every mutation. Approving a domain that's blocked, or
    blocking one that's approved, is handled explicitly (block always
    wins; approving over an existing block is rejected until the block is
    removed).
- **`backend/generation/executor.py` updated** — every generation prompt
  (manual-context, local-doc RAG, or web-sourced RAG — indistinguishable
  once ingested) now wraps its context block via `wrap_context_as_data()`
  before it reaches the LLM, closing the gap Phase 3 left (citation
  tagging existed, but the explicit "treat as data" framing was not yet
  applied at the prompt-assembly layer).
- **Tests** (`test_domain_policy.py`, `test_injection_defense.py`,
  `test_sanitize.py`, `test_web_fetch.py`, `test_web_ingestion.py`,
  `test_api_web_research.py`) — 37 new tests, all passing (128 total),
  covering: deny-by-default and block-overrides-approve domain logic,
  every injection pattern the scrubber targets (verified to redact only
  the offending line, not the whole document), HTML sanitization
  including the comment-hiding case, a monkeypatched `RequestsFetcher`
  proving its content-type/size-cap/error-handling logic without any
  network access, a full fetch→sanitize→scrub→chunk→embed→store flow
  against a simulated malicious page (confirming the injection payload
  never reaches a stored chunk while legitimate content survives), and a
  full API flow (reject unapproved URL → approve domain → ingest
  succeeds → block reverses it → search-preview annotates without
  ingesting → RBAC blocks non-administrators from domain management).

```
$ python3 -m pytest tests/ -q
............................................................................
........................................................
128 passed in 15.14s
```

## 2. Design decisions

- **Deny-by-default is structural, not a default parameter.**
  `DomainPolicy.evaluate()` has exactly one path that returns
  `allowed=True`, gated on an `ApprovedDomain` row actually existing —
  there's no "if list is empty, allow everything" fallback anywhere, which
  is the kind of thing that's easy to accidentally introduce as a
  "helpful" dev convenience and exactly what the master prompt's "nothing
  is auto-approved" rule forbids.
- **No domains are pre-seeded in code.** `docs/PHASE0-DESIGN.md` section 8
  suggested a "small curated seed list" of OER/standards-body domains, but
  hardcoding specific real external domains as pre-approved in application
  code would assert an authority judgment the codebase isn't positioned to
  make for every institution. Seeding the approved list is a per-deployment
  administrative action via `POST /api/domains/approved`, not a code
  default — this also keeps `DomainPolicy`'s "nothing is auto-approved"
  guarantee true from a completely fresh install.
- **Upload and URL ingestion share one pipeline (`ingest_pages()`), not
  two.** Before Phase 4, `rag/ingestion.py`'s `ingest_document()` did
  extraction and chunk/embed/store in one function; splitting that let
  `rag/web/ingestion.py` reuse the exact same chunk/embed/store logic
  after its own fetch/sanitize/scrub steps, so retrieval never needs to
  know or care whether a chunk came from an upload or the web.
- **The injection scrub redacts lines, not documents.** A single
  suspicious sentence embedded in an otherwise-legitimate OER page
  shouldn't discard the whole page's genuine educational content — but it
  also must never survive into a stored chunk. Line-level granularity is
  the balance point; `injection_findings` is still surfaced (via the API
  response and the audit log) so a human can review what was caught,
  rather than the redaction happening silently.
- **`wrap_context_as_data()` now applies to every generation prompt, not
  just Phase 4 ones.** Manual context (Phase 2) can also be pasted by an
  instructor from an untrusted source; there's no principled reason to
  apply the "treat as data" framing only to RAG-retrieved content. This
  was a small change to `_generate_one()`'s prompt assembly, not new
  architecture.
- **`SearXNGSearchAdapter` over a commercial search API**, per Phase 0's
  self-hosting preference — avoids sending query traffic (which can leak
  what topics/subjects a university is building question banks for) to a
  third party, and avoids an external API-key dependency for something
  that's supplementary discovery, not the enforcement boundary (the
  enforcement boundary is `DomainPolicy`, checked regardless of how a URL
  was discovered).
- **`search-preview` never ingests.** Splitting "find candidate URLs" from
  "ingest this specific URL" into two endpoints means a search result
  landing in the system always goes through the same explicit,
  audited `POST /api/sources/from-url` call and the same domain check —
  there's no shortcut where a search hit is trusted more than a
  hand-typed URL.

## 3. Configuration

New environment variables (all prefixed `AIQBE_`, see `backend/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `AIQBE_WEB_FETCH_TIMEOUT_SECONDS` | `15.0` | Per-request timeout for `RequestsFetcher`. |
| `AIQBE_WEB_FETCH_MAX_BYTES` | `20971520` (20MB) | Streamed download cap, enforced mid-download. |
| `AIQBE_WEB_FETCH_RESPECT_ROBOTS_TXT` | `true` | Set false only for an institution's own domains where this is known unnecessary. |
| `AIQBE_SEARXNG_BASE_URL` | unset | Set to a self-hosted SearXNG instance to enable `search-preview`; unset means `NullSearchAdapter` (no search, direct URL ingestion still works). |

Managing approved domains:

```bash
curl -X POST http://localhost:8000/api/domains/approved \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"domain": "oer.university.example", "priority": 100, "min_quality_score": 0.7}'
```

Ingesting an approved URL:

```bash
curl -X POST http://localhost:8000/api/sources/from-url \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"url": "https://oer.university.example/physics/newtons-laws", "external_subject_id": "phys-1", "external_topic_id": "newtons-laws"}'
```

## 4. Limitations

- **No live Internet fetch has been performed** — see the status line
  above. `RequestsFetcher` is written and its logic unit-tested via a
  monkeypatched transport, but running it against a real approved domain
  needs a network policy that permits it (this sandbox's blocks
  everything). This mirrors Phase 1's LLM benchmark and Phase 3's
  embedding model exactly.
- **`min_quality_score` on `ApprovedDomain` is stored but not yet
  enforced anywhere.** There is no automated content-quality scoring in
  this phase to compare against it — Phase 0 flags this as a policy
  field, and until a quality signal exists to feed it (possibly part of
  Phase 7's grounding/quality validators), it's metadata an administrator
  can set and read but the pipeline doesn't yet act on.
- **`robots.txt` handling is best-effort and fails open on fetch
  failure** (can't reach `/robots.txt` → treated as allowed), which is
  the common convention but worth stating plainly rather than implying
  stricter enforcement than exists.
- **No copyright/license classification beyond what's manually
  recorded.** `AcademicSource.license` exists but Phase 4 doesn't attempt
  to detect or verify a page's actual license — matching master prompt
  section 36's instruction to prefer OER/institution-owned/properly
  licensed sources, but the burden of confirming that is still on the
  administrator approving a domain, not automated here.
- **Only `text/html` and `application/pdf` are fetchable** — the same
  format restriction as Phase 3's upload endpoint, for the same reason
  (an unbounded format list is an unbounded parsing-vulnerability
  surface).
- **The injection-defense regex list is a fixed, documented set of known
  patterns, not a learned or exhaustive classifier.** It reduces exposure
  to known phrasings; it is not proof against a novel injection attempt,
  which is exactly why `wrap_context_as_data()` exists as an independent
  second layer rather than the regex scrub being the only defense.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "Approved-domain-gated retrieval merges into the same knowledge pack;
> prompt-injection test suite (malicious/instructional web content)
> passes without behavior change."

**Met**, within the limitations above: web-sourced chunks merge into the
same `document_chunks` table and Topic Knowledge Pack retrieval Phase 3
built, with zero executor changes needed for that merge; a simulated
malicious-page injection test confirms the payload is scrubbed before
storage while legitimate content survives; domain approval is enforced
before any fetch occurs. What's not yet demonstrated is a fetch against
the live, real Internet — that requires a network policy this sandbox
doesn't have, the same gap Phase 1 and Phase 3 already carry forward.
