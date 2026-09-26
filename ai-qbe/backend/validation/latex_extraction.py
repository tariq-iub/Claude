"""Extracts LaTeX/mhchem math segments from mixed prose+notation text.

Per docs/PHASE0-DESIGN.md section 16, source text stores notation inline
as MathJax-compatible delimited LaTeX (e.g. "the equation $F=ma$ shows...")
rather than pre-rendered images. Before that text can be checked for
malformed notation (backend/validation/notation.py) or rendered, the math
segments need to be pulled out from the surrounding prose -- MathJax's own
TeX processor expects a single math expression per call, not a whole
paragraph of mixed text.
"""

from __future__ import annotations

import dataclasses
import re

# Ordered so the two-character display-math delimiters ($$, \[) are tried
# before the one-character/two-character inline ones ($, \() -- otherwise
# "$$x$$" would be misparsed as two empty inline "$$" pairs.
_DELIMITER_PATTERNS = [
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), "display"),
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), "display"),
    (re.compile(r"\\\((.+?)\\\)", re.DOTALL), "inline"),
    (re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL), "inline"),
]


@dataclasses.dataclass
class MathSegment:
    latex: str
    display: bool
    start: int
    end: int


def extract_math_segments(text: str) -> list[MathSegment]:
    """Finds all delimited math segments in `text`. Overlapping matches
    (e.g. a `$$...$$` span that would also match the inline `$...$`
    pattern) are resolved by processing longer/display delimiters first
    and excluding any text range already claimed.
    """
    claimed: list[tuple[int, int]] = []
    segments: list[MathSegment] = []

    for pattern, kind in _DELIMITER_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            segments.append(
                MathSegment(latex=match.group(1).strip(), display=(kind == "display"), start=start, end=end)
            )

    segments.sort(key=lambda s: s.start)
    return segments


def has_math_notation(text: str) -> bool:
    return len(extract_math_segments(text)) > 0
