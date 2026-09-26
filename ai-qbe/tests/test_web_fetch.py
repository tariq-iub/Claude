from rag.web.fetch import FetchResult, RequestsFetcher, StaticFetcher


def test_static_fetcher_returns_configured_response():
    result = FetchResult("https://approved.example/page", 200, "text/html", b"<html>hi</html>")
    fetcher = StaticFetcher({"https://approved.example/page": result})
    fetched = fetcher.fetch("https://approved.example/page")
    assert fetched.ok
    assert fetched.body == b"<html>hi</html>"


def test_static_fetcher_returns_not_found_for_unknown_url():
    fetcher = StaticFetcher({})
    fetched = fetcher.fetch("https://unknown.example/x")
    assert not fetched.ok
    assert fetched.rejected_reason == "not_found_in_static_fetcher"


def test_fetch_result_ok_requires_no_rejection_reason_and_2xx():
    assert FetchResult("u", 200, "text/html", b"x").ok
    assert not FetchResult("u", 404, "text/html", b"x").ok
    assert not FetchResult("u", 200, "text/html", b"x", rejected_reason="disallowed_content_type").ok


class _FakeResponse:
    def __init__(self, status_code, headers, chunks):
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        pass


def test_requests_fetcher_rejects_disallowed_content_type(monkeypatch):
    import requests

    def fake_get(url, timeout, headers, stream=False):
        return _FakeResponse(200, {"Content-Type": "application/msword"}, [b"data"])

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = RequestsFetcher(respect_robots_txt=False)
    result = fetcher.fetch("https://approved.example/doc.docx")
    assert not result.ok
    assert "disallowed_content_type" in result.rejected_reason


def test_requests_fetcher_enforces_max_bytes(monkeypatch):
    import requests

    big_chunks = [b"x" * 1000 for _ in range(20)]  # 20,000 bytes

    def fake_get(url, timeout, headers, stream=False):
        return _FakeResponse(200, {"Content-Type": "text/html"}, big_chunks)

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = RequestsFetcher(max_bytes=5000, respect_robots_txt=False)
    result = fetcher.fetch("https://approved.example/huge-page")
    assert not result.ok
    assert result.rejected_reason == "exceeds_max_bytes"


def test_requests_fetcher_accepts_html_under_limit(monkeypatch):
    import requests

    def fake_get(url, timeout, headers, stream=False):
        return _FakeResponse(200, {"Content-Type": "text/html; charset=utf-8"}, [b"<html>ok</html>"])

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = RequestsFetcher(respect_robots_txt=False)
    result = fetcher.fetch("https://approved.example/page")
    assert result.ok
    assert result.content_type == "text/html"
    assert result.body == b"<html>ok</html>"


def test_requests_fetcher_network_failure_is_reported_not_raised(monkeypatch):
    import requests

    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = RequestsFetcher(respect_robots_txt=False)
    result = fetcher.fetch("https://approved.example/page")
    assert not result.ok
    assert "request_failed" in result.rejected_reason
