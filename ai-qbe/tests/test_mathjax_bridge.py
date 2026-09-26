"""Tests backend/validation/mathjax_bridge.py against the REAL MathJax
library (mathjax-full, via tools/mathjax_validator/render.mjs) -- not a
mock. This is the one Phase 6 component that can be genuinely,
end-to-end verified in this sandbox: unlike an LLM or a hosted embedding
model, MathJax's rendering behavior needs no network access once
`npm install` has fetched the package (see
tools/mathjax_validator/package.json), and Node + that dependency are
both installed in this environment.

If Node/mathjax-full are ever unavailable in a given environment, these
tests are skipped rather than failing -- see the module-level skipif.
"""

import pytest

from backend.validation.mathjax_bridge import MathJaxUnavailableError, is_available, render_check_batch

pytestmark = pytest.mark.skipif(
    not is_available(), reason="Node.js + tools/mathjax_validator's mathjax-full dependency not installed"
)


def test_valid_algebra_renders_ok():
    results = render_check_batch(["x^2 + y^2 = r^2"])
    assert results[0].ok


def test_missing_close_brace_is_a_real_render_error():
    results = render_check_batch(["\\frac{a}{b"])
    assert not results[0].ok
    assert "brace" in results[0].error.lower()


def test_undefined_command_is_a_real_render_error():
    results = render_check_batch(["\\notarealcommand{x}"])
    assert not results[0].ok
    assert "undefined" in results[0].error.lower() or "control sequence" in results[0].error.lower()


def test_mhchem_reaction_renders_ok():
    results = render_check_batch(["\\ce{2H2 + O2 -> 2H2O}"])
    assert results[0].ok


def test_mhchem_ion_charge_renders_ok():
    results = render_check_batch(["\\ce{SO4^2-}"])
    assert results[0].ok


def test_vectors_and_greek_letters_render_ok():
    results = render_check_batch(["\\vec{A} + \\vec{B} = \\alpha \\hat{n}"])
    assert results[0].ok


def test_matrix_environment_renders_ok():
    results = render_check_batch(["\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}"])
    assert results[0].ok


def test_calculus_notation_renders_ok():
    results = render_check_batch(
        ["\\int_0^1 x^2\\,dx", "\\lim_{x\\to 0} \\frac{\\sin x}{x}", "\\sum_{i=1}^n i^2"]
    )
    assert all(r.ok for r in results)


def test_scientific_notation_renders_ok():
    results = render_check_batch(["6.022 \\times 10^{23}"])
    assert results[0].ok


def test_batch_preserves_order_and_mixes_valid_invalid():
    results = render_check_batch(["x^2", "\\badcmd{y}", "z^3"])
    assert [r.ok for r in results] == [True, False, True]


def test_empty_batch_returns_empty_list():
    assert render_check_batch([]) == []


def test_unbalanced_brackets_in_matrix_is_an_error():
    results = render_check_batch(["\\begin{bmatrix} a & b \\end{matrx}"])
    assert not results[0].ok
