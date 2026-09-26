"""API-level test for Phase 7: quality scores and QA validation results
(answer verification, dedup, distractor) are visible through the
question-review endpoints, not just internal to the executor.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import User
from backend.security.auth import hash_password


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_qa.db"
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


def test_question_detail_exposes_quality_score_and_qa_validation_results(client):
    token = _login(client)
    payload = {
        "external_subject_id": "phys-1",
        "requested_count": 1,
        "difficulty_distribution": {"easy": 1.0},
        "bloom_distribution": {"remember": 1.0},
        "question_type_distribution": {"single_best_answer": 1.0},
        "topics": [{"external_topic_id": "newtons-laws", "manual_context": "F=ma.", "weight": 1.0}],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=payload, headers=_auth(token))
    job_id = resp.json()["id"]
    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth(token))
    assert resp.status_code == 200, resp.text

    resp = client.get(f"/api/questions?generation_job_id={job_id}", headers=_auth(token))
    questions = resp.json()
    assert len(questions) >= 1

    pending = [q for q in questions if q["status"] == "PENDING_REVIEW"]
    assert pending, "expected at least one candidate to reach PENDING_REVIEW"
    question = pending[0]

    assert question["quality_score"] is not None
    assert 0.0 <= question["quality_score"]["composite_score"] <= 100.0

    validator_names = {v["validator_name"] for v in question["validation_results"]}
    assert {"structural", "notation", "dedup", "distractor", "answer_verifier", "difficulty_bloom"}.issubset(
        validator_names
    )

    resp = client.get(f"/api/questions/{question['id']}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["quality_score"]["composite_score"] == question["quality_score"]["composite_score"]
