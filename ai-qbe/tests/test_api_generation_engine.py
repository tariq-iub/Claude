"""API-level tests for Phase 5: custom question_type_distribution on job
creation, and the single-item regenerate endpoint (which uses the
non-batch _generate_one path).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import User
from backend.security.auth import hash_password


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_engine.db"
    app = create_app(database_url=f"sqlite:///{db_path}", seed_admin=False)

    from backend.api import deps

    db = deps._session_factory()
    db.add(User(username="admin", hashed_password=hash_password("adminpass"), role="administrator"))
    db.add(User(username="reviewer1", hashed_password=hash_password("reviewpass"), role="academic_reviewer"))
    db.commit()
    db.close()

    return TestClient(app)


def _login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_job_creation_rejects_invalid_question_type_key(client):
    token = _login(client, "admin", "adminpass")
    payload = {
        "external_subject_id": "phys-1",
        "requested_count": 4,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "question_type_distribution": {"not_a_real_type": 1.0},
        "topics": [{"external_topic_id": "newtons-laws", "manual_context": "F=ma.", "weight": 1.0}],
    }
    resp = client.post("/api/generation-jobs", json=payload, headers=_auth(token))
    assert resp.status_code == 422


def test_job_creation_with_custom_question_type_distribution_produces_matching_candidates(client):
    token = _login(client, "admin", "adminpass")
    payload = {
        "external_subject_id": "phys-1",
        "requested_count": 20,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "question_type_distribution": {"scenario_based": 1.0},
        "topics": [{"external_topic_id": "newtons-laws", "manual_context": "F=ma.", "weight": 1.0}],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=payload, headers=_auth(token))
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]

    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth(token))
    assert resp.status_code == 200, resp.text

    resp = client.get(f"/api/questions?generation_job_id={job_id}", headers=_auth(token))
    questions = resp.json()
    assert len(questions) > 0
    assert all(q["question_type"] == "scenario_based" for q in questions)


def test_regenerate_produces_a_fresh_candidate_and_supersedes_the_old_one(client):
    admin = _login(client, "admin", "adminpass")
    reviewer = _login(client, "reviewer1", "reviewpass")

    payload = {
        "external_subject_id": "phys-1",
        "requested_count": 1,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        # Deliberately non-default, so a regenerate implementation that
        # silently resets question_type to the PlanCell default would be
        # caught by the assertion below.
        "question_type_distribution": {"negative": 1.0},
        "topics": [{"external_topic_id": "newtons-laws", "manual_context": "F=ma.", "weight": 1.0}],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=payload, headers=_auth(admin))
    job_id = resp.json()["id"]
    client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth(admin))

    resp = client.get(f"/api/questions?generation_job_id={job_id}", headers=_auth(admin))
    questions = resp.json()
    assert len(questions) >= 1
    original_id = questions[0]["id"]
    assert questions[0]["question_type"] == "negative"

    resp = client.post(f"/api/questions/{original_id}/regenerate", headers=_auth(reviewer))
    assert resp.status_code == 200, resp.text
    new_question = resp.json()
    assert new_question["id"] != original_id
    assert new_question["status"] == "PENDING_REVIEW"
    assert new_question["question_type"] == "negative"  # preserved from the superseded candidate

    resp = client.get(f"/api/questions/{original_id}", headers=_auth(admin))
    assert resp.json()["status"] == "REJECTED"  # superseded, not silently left as PENDING_REVIEW
