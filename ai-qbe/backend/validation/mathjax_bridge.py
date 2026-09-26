"""Python <-> real-MathJax bridge.

Shells out to tools/mathjax_validator/render.mjs (Node + mathjax-full,
MathJax's own server-side rendering library) to answer "does this LaTeX
actually render in MathJax" with the real thing, not an approximation.
All snippets from one validation call are batched into a single Node
process invocation to amortize Node/MathJax startup cost (~200-400ms)
across however many math segments one MCQ candidate contains, rather than
spawning a process per segment.

If Node or the `tools/mathjax_validator` npm dependencies aren't
installed, this raises `MathJaxUnavailableError` rather than silently
treating every expression as valid -- notation.py's caller must decide
what "we couldn't check" means for its status transition, and it must
never be conflated with "we checked and it passed."
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

_VALIDATOR_DIR = Path(__file__).resolve().parents[2] / "tools" / "mathjax_validator"
_RENDER_SCRIPT = _VALIDATOR_DIR / "render.mjs"


class MathJaxUnavailableError(RuntimeError):
    pass


@dataclasses.dataclass
class RenderCheckResult:
    ok: bool
    error: str | None = None


def is_available() -> bool:
    return (
        shutil.which("node") is not None
        and _RENDER_SCRIPT.exists()
        and (_VALIDATOR_DIR / "node_modules" / "mathjax-full").exists()
    )


def render_check_batch(latex_snippets: list[str], *, timeout_seconds: float = 30.0) -> list[RenderCheckResult]:
    """Renders each of `latex_snippets` through real MathJax and reports
    whether it rendered without error. Order-preserving, one result per
    input snippet.
    """
    if not latex_snippets:
        return []
    if not is_available():
        raise MathJaxUnavailableError(
            "Node.js + tools/mathjax_validator's mathjax-full dependency are not "
            "installed. Run `npm install` in tools/mathjax_validator/ (see "
            "docs/PHASE6-SCIENTIFIC-CONTENT.md)."
        )

    try:
        proc = subprocess.run(
            ["node", str(_RENDER_SCRIPT)],
            input=json.dumps(latex_snippets),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(_VALIDATOR_DIR),
        )
    except subprocess.TimeoutExpired as exc:
        raise MathJaxUnavailableError(f"MathJax render subprocess timed out: {exc}") from exc

    if proc.returncode != 0:
        raise MathJaxUnavailableError(f"MathJax render subprocess failed: {proc.stderr[:2000]}")

    try:
        raw_results = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MathJaxUnavailableError(f"MathJax render subprocess returned invalid JSON: {exc}") from exc

    return [RenderCheckResult(ok=r["ok"], error=r.get("error")) for r in raw_results]
