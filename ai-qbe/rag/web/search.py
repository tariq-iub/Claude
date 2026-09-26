"""Academic web search adapter (docs/PHASE0-DESIGN.md section 8).

`SearXNGSearchAdapter` is the self-hosted default (a SearXNG instance the
institution runs itself, avoiding a dependency on a commercial search
API and keeping query traffic internal). `NullSearchAdapter` is the
dev/test default when no search backend is configured -- returning no
results rather than fabricating any, consistent with "never fabricate
references" (master prompt section 52): a disabled/unconfigured search
adapter must fail closed (no candidate URLs), not silently invent some.

Search results are only ever candidate URLs -- they still go through
DomainPolicy before anything is fetched (rag/web/domain_policy.py), so a
search adapter returning an unapproved domain is not itself a policy
violation; the fetch step is where enforcement happens.
"""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


class ISearchAdapter(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        ...


class NullSearchAdapter(ISearchAdapter):
    def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        return []


class SearXNGSearchAdapter(ISearchAdapter):
    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        import requests

        resp = requests.get(
            f"{self.base_url}/search",
            params={"q": query, "format": "json"},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json()
        results = []
        for item in body.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                )
            )
        return results
