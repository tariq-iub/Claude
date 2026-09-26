"""Clean -> normalize -> semantic chunking, per docs/PHASE0-DESIGN.md
section 10.

Chunking is heading/paragraph-boundary aware rather than a fixed
character-width split, specifically so a formula and the sentence
explaining it aren't torn across two chunks. Word count is used as a
token-count approximation (~0.75 words per token is the usual English
rule of thumb, so a 400-word target chunk is roughly 500 tokens) --
good enough for chunk-sizing purposes without pulling in a model-specific
tokenizer this early in the pipeline.
"""

from __future__ import annotations

import dataclasses
import re

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^(?=.{2,80}$)(?:[A-Z][A-Za-z0-9 ,'\-]*|[0-9]+(\.[0-9]+)*\s+.+)$")


@dataclasses.dataclass
class Chunk:
    index: int
    text: str
    page: int | None
    section: str | None
    word_count: int


def clean_text(text: str) -> str:
    """Strips boilerplate (standalone page-number lines), normalizes
    whitespace and line endings. Deliberately conservative -- it does not
    try to detect/strip repeated running headers/footers across pages,
    which needs cross-page comparison the per-page extraction API doesn't
    expose yet; flagged here rather than silently pretending to handle it.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in text.split("\n") if not _PAGE_NUMBER_LINE_RE.match(line)]
    text = "\n".join(lines)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _looks_like_heading(paragraph: str) -> bool:
    line = paragraph.strip()
    return bool(line) and "\n" not in line and bool(_HEADING_RE.match(line)) and not line.endswith((".", ",", ";"))


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces, current, current_words = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > max_words:
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def semantic_chunk(
    text: str, *, page: int | None = None, max_words: int = 400, min_words: int = 40
) -> list[Chunk]:
    """Packs paragraphs into chunks up to `max_words`, tracking the most
    recent heading-like line as the chunk's `section` label, and merging
    trailing under-sized chunks (< min_words) into the previous one so a
    stray short paragraph at a page boundary doesn't become its own
    near-empty chunk.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_words = 0
    current_section: str | None = None

    def flush():
        nonlocal current_parts, current_words
        if not current_parts:
            return
        joined = "\n\n".join(current_parts)
        chunks.append(Chunk(index=len(chunks), text=joined, page=page, section=current_section, word_count=current_words))
        current_parts, current_words = [], 0

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            flush()
            current_section = paragraph
            continue

        word_count = len(paragraph.split())
        if word_count > max_words:
            flush()
            for piece in _split_long_paragraph(paragraph, max_words):
                chunks.append(
                    Chunk(index=len(chunks), text=piece, page=page, section=current_section, word_count=len(piece.split()))
                )
            continue

        if current_words + word_count > max_words:
            flush()
        current_parts.append(paragraph)
        current_words += word_count

    flush()

    # Merge an undersized trailing chunk into its predecessor within the
    # same section, rather than leaving a near-empty final chunk.
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and chunk.word_count < min_words and merged[-1].section == chunk.section:
            prev = merged[-1]
            merged[-1] = Chunk(
                index=prev.index,
                text=f"{prev.text}\n\n{chunk.text}",
                page=prev.page,
                section=prev.section,
                word_count=prev.word_count + chunk.word_count,
            )
        else:
            merged.append(chunk)

    for i, chunk in enumerate(merged):
        chunk.index = i
    return merged
