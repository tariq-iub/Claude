"""End-to-end API test for Phase 4: domain management, URL ingestion
(approved vs. rejected), search preview, and RBAC (Administrator-only
domain management).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import User
from backend.security.auth import hash_password
from rag.web.fetch import FetchResult, StaticFetcher
from rag.web.search import ISearchAdapter, SearchResult

HTML_PAGE = (
    "<html><head><title>Vectors - OER</title></head><body>"
    "<p>Vectors have both magnitude and direction, unlike scalar quantities.</p>"
    "</body></html>"
)


class FakeSearchAdapter(ISearchAdapter):
    def search(self, query, *, max_results=10):
        return [
            SearchResult(url="https://oer.example.edu/vectors", title="Vectors - OER", snippet="..."),
            SearchResult(url="https://random-blog.example/vectors", title="My blog about vectors", snippet="..."),
        ]


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_web.db"
    url = "https://oer.example.edu/vectors"
    fetcher = StaticFetcher({url: FetchResult(url, 200, "text/html", HTML_PAGE.encode())})
    app = create_app(
        database_url=f"sqlite:///{db_path}",
        seed_admin=False,
        web_fetcher=fetcher,
        search_adapter=FakeSearchAdapter(),
    )

    from backend.api import deps

    db = deps._session_factory()
    db.add(User(username="admin", hashed_password=hash_password("adminpass"), role="administrator"))
    db.add(User(username="gen1", hashed_password=hash_password("genpass"), role="question_generator"))
    db.commit()
    db.close()

    return TestClient(app)


def _login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_url_ingestion_rejected_until_domain_approved(client):
    admin = _login(client, "admin", "adminpass")

    resp = client.post(
        "/api/sources/from-url",
        json={"url": "https://oer.example.edu/vectors", "external_subject_id": "phys-1", "external_topic_id": "vectors"},
        headers=_auth(admin),
    )
    assert resp.status_code == 403
    assert "domain_not_in_approved_list" in resp.text

    resp = client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))
    assert resp.status_code == 201

    resp = client.post(
        "/api/sources/from-url",
        json={"url": "https://oer.example.edu/vectors", "external_subject_id": "phys-1", "external_topic_id": "vectors"},
        headers=_auth(admin),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["chunk_count"] >= 1
    assert body["url"] == "https://oer.example.edu/vectors"


def test_only_administrator_can_manage_domains(client):
    gen_token = _login(client, "gen1", "genpass")
    resp = client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(gen_token))
    assert resp.status_code == 403

    admin = _login(client, "admin", "adminpass")
    resp = client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))
    assert resp.status_code == 201

    # Both roles can read the list
    resp = client.get("/api/domains/approved", headers=_auth(gen_token))
    assert resp.status_code == 200
    assert any(d["domain"] == "oer.example.edu" for d in resp.json())


def test_blocking_a_domain_removes_it_from_approved_and_denies_future_ingestion(client):
    admin = _login(client, "admin", "adminpass")
    client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))

    resp = client.post("/api/domains/blocked", json={"domain": "oer.example.edu", "reason": "license revoked"}, headers=_auth(admin))
    assert resp.status_code == 201

    resp = client.get("/api/domains/approved", headers=_auth(admin))
    assert not any(d["domain"] == "oer.example.edu" for d in resp.json())

    resp = client.post(
        "/api/sources/from-url",
        json={"url": "https://oer.example.edu/vectors", "external_subject_id": "phys-1", "external_topic_id": "vectors"},
        headers=_auth(admin),
    )
    assert resp.status_code == 403
    assert "domain_blocked" in resp.text


def test_search_preview_annotates_domain_approval_without_ingesting(client):
    admin = _login(client, "admin", "adminpass")
    client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))

    resp = client.post("/api/sources/search-preview", json={"query": "vectors"}, headers=_auth(admin))
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2
    by_url = {r["url"]: r for r in results}
    assert by_url["https://oer.example.edu/vectors"]["domain_decision"] == "approved"
    assert by_url["https://random-blog.example/vectors"]["domain_decision"] == "domain_not_in_approved_list"

    # search-preview must never ingest anything by itself
    resp = client.get("/api/sources/1", headers=_auth(admin))
    assert resp.status_code == 404


def test_duplicate_approve_and_missing_unblock_are_rejected_cleanly(client):
    admin = _login(client, "admin", "adminpass")
    client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))
    resp = client.post("/api/domains/approved", json={"domain": "oer.example.edu"}, headers=_auth(admin))
    assert resp.status_code == 409

    resp = client.delete("/api/domains/blocked/never-blocked.example", headers=_auth(admin))
    assert resp.status_code == 404
