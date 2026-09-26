# AI-QBE — Phase 6: Scientific Content Support

Status: **Implemented and tested against real MathJax (215/215 tests
passing).** Unlike every previous phase touching an external system, this
one required no compromise: MathJax needs no network access to check
notation once its npm package is installed, and Node + that dependency
are both available in this sandbox, so "renders correctly in MathJax" is
verified against the actual library, not an approximation.

## 1. What was implemented

- **`tools/mathjax_validator/`** — a small Node.js subprocess
  (`render.mjs`) built on `mathjax-full` (MathJax's own server-side
  rendering library, the same TeX parser that will render questions for
  real in the Phase 8 review UI). It renders each LaTeX/mhchem snippet to
  SVG via a headless "lite" DOM adaptor (no browser) and reports whether
  a `data-mjx-error` node appears in the output. Package selection is
  `AllPackages` minus `noundefined` — deliberately, since `noundefined`
  makes MathJax silently render an unrecognized macro as inert red text
  instead of raising a real, catchable error, which is exactly the
  failure mode this validator exists to catch (master prompt section 19:
  "unsupported MathJax commands"). Every other package (`mhchem`,
  `physics`, `ams` matrix/array environments, `cancel`, `bbox`, etc.)
  stays enabled.
- **`backend/validation/mathjax_bridge.py`** — the Python↔Node bridge.
  Batches every math segment from one validation call into a single Node
  process invocation (amortizing the ~200-400ms Node/MathJax startup cost
  across however many `$...$` segments one MCQ contains, rather than
  spawning a process per segment). Raises `MathJaxUnavailableError`
  explicitly if Node or the npm dependency aren't installed — this is
  never silently treated as "passed" anywhere downstream.
- **`backend/validation/latex_extraction.py`** — pulls delimited math
  segments (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) out of mixed
  prose+notation text, since MathJax's TeX processor expects one
  expression per call, not a whole paragraph (docs/PHASE0-DESIGN.md
  section 16: notation is stored inline as delimited LaTeX text, never
  pre-rendered to images).
- **`backend/validation/chemistry.py`** — content-level chemistry
  validation, independent of LaTeX syntax: a formula can be syntactically
  valid mhchem while still being chemical nonsense.
  - `validate_formula()`: parses `H2O`, `Ca(OH)2`, `\ce{SO4^2-}`-style
    notation against the full 118-element IUPAC symbol table, catching
    unknown symbols and unbalanced parentheses.
  - `validate_equation_balance()`: parses a reaction equation's two
    sides and checks equal atom counts per element — this is what catches
    the classic `H2 + O2 -> H2O` (unbalanced) case that pure LaTeX
    rendering would never flag, since it's syntactically fine notation.
  - **Documented scope decision**: a bare digit immediately before a
    charge sign with no caret (`Fe3+`) is parsed as the *preceding
    element's subscript*, not charge magnitude — this is what correctly
    handles the far more common `NH4+` (ammonium: subscript 4 on H,
    charge +1) the same way, without a chemistry-aware valence lookup.
    Charge magnitude greater than 1 is expected to use the explicit caret
    form (`\ce{Fe^3+}`), matching standard mhchem/IUPAC practice.
- **`backend/validation/units.py`** — the "lightweight regex/lookup"
  physics-unit checker Phase 0 section 13 calls for: flags a numeric
  option that looks like it's missing a recognized SI/derived unit.
  Deliberately a soft signal (warnings, never a failure) — it does not
  attempt dimensional analysis across options, matching the explicitly
  narrow scope in the design.
- **`backend/validation/notation.py`** — orchestrates all three: extracts
  math segments from the question/options/explanation, batches them
  through MathJax, runs chemistry content checks on any `\ce{...}`
  segments regardless of MathJax availability (pure Python, no
  subprocess needed), and collects unit warnings. Exposes a
  `status_label` property (`pass`/`fail`/`uncertain`) mirroring the
  PASS/FAIL/UNCERTAIN pattern from the answer-verification schema: a
  candidate containing math that couldn't be checked (MathJax
  unavailable) is `uncertain`, never silently recorded the same as a real
  `pass`.
- **Wired into the generation executor** — `_apply_parsed_item()` now
  runs `validate_notation()` immediately after structural validation
  passes, recording an `MCQValidationResult` row (`validator_name=
  "notation"`) and marking the candidate `INVALID` (never reaching
  `PENDING_REVIEW`) if notation is malformed — matching master prompt
  section 19's "no malformed notation should automatically enter the
  approved bank," enforced at generation time rather than deferred.
- **Tests** — 74 new tests, all passing (215 total): LaTeX extraction
  edge cases (including the `$$x$$` vs. two empty `$...$` pairs
  ambiguity), 12 tests running real MathJax renders (valid algebra,
  calculus, vectors, Greek letters, matrices, scientific notation,
  mhchem reactions and ions, and multiple flavors of malformed LaTeX), 19
  chemistry tests (balanced/unbalanced equations including the classic
  combustion case, formula parsing, the documented `Fe3+`/`NH4+` scope
  decision made explicit as its own test), 9 unit-detection tests, a
  **22-case seeded valid/malformed acceptance test**
  (`test_notation_validation.py`) directly exercising the Phase 0
  acceptance criterion, and 3 executor-integration tests proving
  malformed LaTeX and an unbalanced chemical equation each actually
  block a candidate from reaching human review end-to-end (not just at
  the unit level).

```
$ python3 -m pytest tests/ -q
........................................................................
........................................................................
.......................................................................
215 passed in 29.79s
```

## 2. Design decisions

- **A real subprocess to real MathJax, not a hand-rolled LaTeX
  syntax checker.** There is no maintained, complete Python port of
  MathJax's TeX parser. A regex-based approximation would inevitably
  diverge from what actually renders in the Phase 8 review UI (and any
  future student-facing exam UI) — exactly the "validator disagrees with
  reality" failure this project's own "never claim unmeasured/unverified
  things" ethos argues against. Batching amortizes the subprocess cost;
  `MathJaxUnavailableError` makes the failure mode explicit rather than
  silently degrading to "everything passes."
- **`noundefined` is deliberately excluded from the package set.**
  Leaving it in (as `AllPackages` does by default) would make MathJax
  treat an unrecognized command as harmless red text instead of a real,
  catchable error — silently defeating the exact check master prompt
  section 19 asks for.
- **Chemistry content validation is independent of, and runs regardless
  of, MathJax availability.** `validate_equation_balance()` and
  `validate_formula()` are pure Python parses with no subprocess
  dependency — an environment where Node isn't installed still catches
  an unbalanced reaction, even though it can't catch a syntactically
  malformed LaTeX brace in the same content.
- **The `Fe3+` vs. `NH4+` ambiguity is resolved by convention, not
  guessed at.** A regex alone cannot distinguish "digit is charge
  magnitude" from "digit is subscript" when both immediately precede a
  bare sign with no caret. Rather than pick a heuristic that would
  silently mis-parse one of the two common cases, the digit is always
  treated as the preceding element's subscript (correct for `NH4+`,
  the more common undergraduate case), and magnitude->1 charges are
  documented as requiring the explicit caret form — a real,
  test-documented scope boundary, not a hidden bug.
- **Unit-presence checking never fails a candidate.** Per Phase 0's own
  "lightweight" framing, a false positive here (flagging a legitimately
  unitless numeric option, e.g. a dimensionless ratio or a count) would
  be worse than the value of a hard gate; it's recorded as a warning
  for a human reviewer, not a rejection.
- **Notation validation runs as a hard gate at generation time**, not
  deferred to a separate Phase 7 pass, because Phase 0 section 19 is
  explicit: "no malformed notation should automatically enter the
  approved bank." Since nothing is auto-approved yet anyway (Phase 7's
  independent fact-verification doesn't exist), catching malformed
  notation before `PENDING_REVIEW` means reviewers never waste time on a
  candidate that's already known to be broken.

## 3. Configuration / setup

No new Python dependencies (the whole Python side is stdlib:
`subprocess`, `json`, `re`, `dataclasses`). One new setup step:

```bash
cd ai-qbe/tools/mathjax_validator
npm install
```

`backend/validation/mathjax_bridge.is_available()` checks for Node +
the installed `mathjax-full` package at call time; if either is missing,
notation checks degrade to `uncertain` (for content that actually
contains math) rather than crashing the generation pipeline or silently
passing.

## 4. Limitations

- **`mathjax-full` (v3) is used, not the newer scoped `@mathjax/src` (v4)
  package** — v3 issues a deprecation notice on install but remains the
  stable, widely-used server-side rendering library (it's what MathJax's
  own CLI tools are built on). Migrating to v4 is a contained change
  inside `tools/mathjax_validator/` if/when needed.
- **No dimensional analysis across options** — the units checker flags a
  single option missing a unit; it does not verify that, say, all four
  options in a numeric question share consistent units, or that a
  computed answer's units are dimensionally correct for the operation
  performed. That level of rigor would need to be paired with Phase 7's
  independent answer verification (SymPy-based, per master prompt
  section 22), which already handles the "is this numeric answer
  actually correct" question — units consistency is a natural extension
  of that work, not this phase's narrower notation-syntax scope.
- **Physics unit table is a fixed list, not exhaustive** — uncommon or
  domain-specific units (e.g. `eV`, `atm`, `cal`) aren't recognized yet
  and would currently produce a (soft, non-blocking) false unit-warning.
  Extending `_SI_UNIT_SYMBOLS` is a one-line change once real usage shows
  which units matter most.
- **The `Fe3+`/`NH4+` charge-notation scope decision** (see design
  decisions) means a small category of monatomic-ion charges written
  without a caret and with magnitude > 1 will be mis-parsed as a
  subscript. This is documented and tested, not silently wrong, but a
  real generation pipeline should be told (via the prompt template) to
  prefer the caret form for exactly this reason.
- **No LaTeX auto-repair.** A malformed candidate is marked `INVALID` and
  discarded from the review queue (matching Phase 2's existing structural
  validation behavior) rather than attempting to fix or re-prompt for a
  corrected version — that's within the scope of the already-existing
  `regenerate` endpoint (Phase 5), which a reviewer or an over-generation
  round can invoke, not an automatic retry inside notation validation
  itself.

## 5. Acceptance criteria check (docs/PHASE0-DESIGN.md section 20)

> "Math/physics/chemistry sample set renders correctly in MathJax;
> notation validator rejects a seeded set of malformed LaTeX/chemical-
> formula test cases with zero false negatives on that set."

**Met, verified against real MathJax**: `tests/test_notation_validation.py`
runs a 10-case valid set (algebra, calculus, vectors, Greek letters,
matrices, scientific notation, mhchem reactions and ions) — zero false
positives — against an 8-case malformed set (unbalanced braces in the
stem/option/explanation, an undefined command, a bad matrix environment,
an unbalanced chemical equation, an unknown element symbol, unbalanced
parentheses) — zero false negatives — all passing. `tests/test_mathjax_bridge.py`
additionally verifies the underlying render calls directly against the
real `mathjax-full` library, and `tests/test_executor_notation_integration.py`
confirms the gate actually blocks a candidate end-to-end through the
generation pipeline, not just at the validator-function level.
