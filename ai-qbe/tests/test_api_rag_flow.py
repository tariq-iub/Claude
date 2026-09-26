"""End-to-end API test for Phase 3: upload a document -> create a RAG-
enabled generation job (no manual_context) -> start it -> confirm the
generated questions carry citations back to the ingested chunks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import User
from backend.security.auth import hash_password


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_rag.db"
    app = create_app(database_url=f"sqlite:///{db_path}", seed_admin=False)

    from backend.api import deps

    db = deps._session_factory()
    db.add(User(username="admin", hashed_password=hash_password("adminpass"), role="administrator"))
    db.commit()
    db.close()

    return TestClient(app)


def _login(client: TestClient) -> str:
    resp = client.post("/api/auth/login", data={"username": "admin", "password": "adminpass"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_upload_source_then_generate_with_rag(client):
    token = _login(client)

    content = (
        b"Newton's Second Law\n\n"
        b"Force equals mass times acceleration. The relationship F=ma is central "
        b"to classical mechanics and describes how an object accelerates under "
        b"a net external force."
    )
    resp = client.post(
        "/api/sources",
        headers=_auth(token),
        data={"external_subject_id": "phys-1", "external_topic_id": "newtons-laws", "title": "Physics Notes"},
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert resp.status_code == 201, resp.text
    source = resp.json()
    assert source["chunk_count"] >= 1

    # Uploading identical bytes again is deduplicated, not a new source
    resp2 = client.post(
        "/api/sources",
        headers=_auth(token),
        data={"external_subject_id": "phys-1", "external_topic_id": "newtons-laws", "title": "Physics Notes (dup)"},
        files={"file": ("notes2.txt", content, "text/plain")},
    )
    assert resp2.status_code == 201
    assert resp2.json()["id"] == source["id"]

    resp = client.get(f"/api/sources/{source['id']}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["ingestion_status"] == "ready"

    # Uploading to an unknown subject/topic is rejected
    resp = client.post(
        "/api/sources",
        headers=_auth(token),
        data={"external_subject_id": "phys-1", "external_topic_id": "does-not-exist", "title": "Bad"},
        files={"file": ("x.txt", b"whatever", "text/plain")},
    )
    assert resp.status_code == 404

    # Unsupported file type is rejected
    resp = client.post(
        "/api/sources",
        headers=_auth(token),
        data={"external_subject_id": "phys-1", "title": "Bad type"},
        files={"file": ("x.docx", b"whatever", "application/msword")},
    )
    assert resp.status_code == 415

    # Now create a RAG-enabled job (no manual_context needed)
    job_payload = {
        "external_subject_id": "phys-1",
        "requested_count": 2,
        "option_count": 4,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "use_rag": True,
        "topics": [{"external_topic_id": "newtons-laws", "weight": 1.0}],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=job_payload, headers=_auth(token))
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]

    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    assert resp.json()["generated_count"] == 2

    resp = client.get(f"/api/questions?generation_job_id={job_id}", headers=_auth(token))
    questions = resp.json()
    assert len(questions) == 2
    for q in questions:
        assert len(q["source_chunk_ids"]) >= 1  # grounded in the ingested chunk(s)


def test_job_without_rag_requires_manual_context(client):
    token = _login(client)
    job_payload = {
        "external_subject_id": "phys-1",
        "requested_count": 1,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "use_rag": False,
        "topics": [{"external_topic_id": "newtons-laws", "weight": 1.0}],  # no manual_context!
    }
    resp = client.post("/api/generation-jobs", json=job_payload, headers=_auth(token))
    assert resp.status_code == 422


def test_rag_job_with_no_ingested_documents_falls_back_gracefully(client):
    """A RAG-enabled job for a topic with nothing ingested yet must not
    crash -- it should still run (falling back to empty/manual context),
    producing candidates that a human reviewer can still see (even if
    low-confidence), never a 500.
    """
    token = _login(client)
    job_payload = {
        "external_subject_id": "phys-1",
        "requested_count": 1,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "use_rag": True,
        "topics": [{"external_topic_id": "vectors", "weight": 1.0}],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=job_payload, headers=_auth(token))
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["generated_count"] == 1
