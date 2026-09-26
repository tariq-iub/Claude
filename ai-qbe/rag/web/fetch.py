"""Sandboxed HTTP fetch for approved-domain content, per
docs/PHASE0-DESIGN.md section 8: timeouts, size caps, robots.txt respect,
content-type allowlist. This is the only place in AI-QBE that makes an
outbound request to an arbitrary (domain-policy-approved) URL, which is
exactly why it goes through this one narrow, auditable interface rather
than any code path calling `requests.get()` directly.

`IWebFetcher` is injectable so tests exercise the full policy/sanitize/
ingest pipeline against canned responses without needing real network
access -- this development sandbox's own outbound network policy blocks
arbitrary HTTP fetches entirely (confirmed: a plain request to
example.com is rejected by the proxy with a 403), so `RequestsFetcher`
itself is written but not exercised against the live internet here, the
same situation Phase 1's LLM benchmark and Phase 3's embedding model were
in. Run it for real once deployed with a network policy that allows
fetching the institution's approved domains (see
docs/PHASE0-DESIGN.md section 34: the LLM/vector-store network is
internal-only, but approved-domain web fetch is a deliberate, narrow
exception to that isolation).
"""

from __future__ import annotations

import abc
import dataclasses
import urllib.robotparser
from urllib.parse import urljoin, urlparse

_ALLOWED_CONTENT_TYPES = ("text/html", "application/pdf")
_DEFAULT_USER_AGENT = "AI-QBE-Bot/0.1 (self-hosted academic question-bank generator; see institution admin)"


@dataclasses.dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: str | None
    body: bytes
    rejected_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.rejected_reason is None and 200 <= self.status_code < 300


class IWebFetcher(abc.ABC):
    @abc.abstractmethod
    def fetch(self, url: str) -> FetchResult:
        ...


class RequestsFetcher(IWebFetcher):
    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_bytes: int = 20 * 1024 * 1024,
        user_agent: str = _DEFAULT_USER_AGENT,
        respect_robots_txt: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt

    def _robots_allow(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        import requests

        parsed = urlparse(url)
        robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
        parser = urllib.robotparser.RobotFileParser()
        try:
            resp = requests.get(robots_url, timeout=self.timeout_seconds, headers={"User-Agent": self.user_agent})
            if resp.status_code >= 400:
                return True  # no robots.txt (or unreadable) => default allow
            parser.parse(resp.text.splitlines())
        except Exception:
            return True  # fail open on robots.txt fetch failure, not fail closed on the whole pipeline
        return parser.can_fetch(self.user_agent, url)

    def fetch(self, url: str) -> FetchResult:
        import requests

        if not self._robots_allow(url):
            return FetchResult(url, 0, None, b"", rejected_reason="disallowed_by_robots_txt")

        try:
            resp = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
                stream=True,
            )
        except requests.RequestException as exc:
            return FetchResult(url, 0, None, b"", rejected_reason=f"request_failed: {exc}")

        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type not in _ALLOWED_CONTENT_TYPES:
            resp.close()
            return FetchResult(url, resp.status_code, content_type, b"", rejected_reason=f"disallowed_content_type: {content_type}")

        body = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > self.max_bytes:
                resp.close()
                return FetchResult(url, resp.status_code, content_type, b"", rejected_reason="exceeds_max_bytes")
        resp.close()

        return FetchResult(url, resp.status_code, content_type, bytes(body))


class StaticFetcher(IWebFetcher):
    """Test/dev fetcher returning pre-canned responses keyed by URL --
    used wherever a real network fetch isn't available (this sandbox) or
    isn't desired (deterministic unit tests)."""

    def __init__(self, responses: dict[str, FetchResult]):
        self._responses = responses

    def fetch(self, url: str) -> FetchResult:
        if url in self._responses:
            return self._responses[url]
        return FetchResult(url, 404, None, b"", rejected_reason="not_found_in_static_fetcher")
