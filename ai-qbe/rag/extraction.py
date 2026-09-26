"""Text extraction from uploaded source documents.

Phase 3 scope: PDF and plain text/Markdown (the "instructor-provided
material" case from docs/PHASE0-DESIGN.md Phase 3). Web-page extraction
(readability-style HTML cleanup) is Phase 4 scope, alongside the
approved-domain retrieval pipeline it belongs to.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExtractedPage:
    page: int | None  # None for non-paginated formats (plain text/Markdown)
    text: str


def extract_pdf(file_bytes: bytes) -> list[ExtractedPage]:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(ExtractedPage(page=i + 1, text=text))
    return pages


def extract_plain_text(file_bytes: bytes) -> list[ExtractedPage]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [ExtractedPage(page=None, text=text)]


def extract_text(file_bytes: bytes, mime_type: str) -> list[ExtractedPage]:
    if mime_type == "application/pdf":
        return extract_pdf(file_bytes)
    if mime_type in ("text/plain", "text/markdown"):
        return extract_plain_text(file_bytes)
    raise ValueError(f"Unsupported mime_type for extraction: {mime_type}")
