# MathJax notation validator (subprocess)

A tiny Node.js script that renders LaTeX/mhchem snippets through the real
MathJax library (`mathjax-full`, server-side, no browser) and reports
whether each one rendered without error. Invoked as a subprocess by
`backend/validation/mathjax_bridge.py` — see
`../../docs/PHASE6-SCIENTIFIC-CONTENT.md` for why this exists and how it
fits into the notation validation pipeline.

## Setup

```bash
cd tools/mathjax_validator
npm install
```

That's it — no build step, no network access needed at *check* time (only
at `npm install` time, to fetch the `mathjax-full` package once).

## Manual test

```bash
echo '["x^2+y^2=r^2", "\\frac{a}{b"]' | node render.mjs
# [{"ok":true},{"ok":false,"error":"Missing close brace"}]
```

## Why a subprocess instead of a Python MathJax port

There is no maintained, complete Python port of MathJax's TeX parser —
`mathjax-full` (npm) is the reference implementation and the same code
that will render these questions for real in the Phase 8 review UI and
any student-facing exam UI. Validating with anything else risks a
validator that disagrees with what actually renders. A subprocess call
per validation batch (not per individual LaTeX snippet — see
`mathjax_bridge.render_check_batch`) keeps the ~200-400ms Node/MathJax
startup cost amortized across every math segment in one MCQ candidate.
