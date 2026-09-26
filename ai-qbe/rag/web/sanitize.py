"""HTML sanitization / main-content extraction for fetched web pages.

Strips script/style/nav/header/footer/form markup and returns visible
text plus the page title, per docs/PHASE0-DESIGN.md section 8
("content sanitization"). This runs BEFORE the injection-defense scrub
(rag/web/injection_defense.py) and before chunking -- fetched HTML is
never passed to the LLM or stored verbatim.
"""

from __future__ import annotations

import dataclasses

_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "form", "noscript", "aside", "iframe")


@dataclasses.dataclass
class SanitizedPage:
    title: str | None
    text: str


def sanitize_html(html: str) -> SanitizedPage:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Comments can carry hidden instructions (master prompt section 35:
    # "hidden instructions" in webpage content) -- strip them before text
    # extraction rather than relying on the injection-defense regex pass
    # to catch everything.
    from bs4 import Comment

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return SanitizedPage(title=title, text=text)
