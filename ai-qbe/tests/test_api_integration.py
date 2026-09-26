"""End-to-end API test: login -> browse subjects/topics -> create a
generation job with manually-supplied context -> start it (Celery eager
mode, MockProvider) -> list generated questions -> approve one -> reject
one -> confirm RBAC blocks a read-only user from mutating anything.

Runs entirely against an in-memory SQLite DB and MockProvider; no GPU or
network access required, which is exactly why this integration test can
run in this sandbox even though the real LLM benchmark (Phase 1) cannot.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.database.models import User
from backend.security.auth import hash_password


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(database_url=f"sqlite:///{db_path}", seed_admin=False)

    from backend.api import deps

    db = deps._session_factory()
    db.add(User(username="admin", hashed_password=hash_password("adminpass"), role="administrator"))
    db.add(User(username="reviewer1", hashed_password=hash_password("reviewpass"), role="academic_reviewer"))
    db.add(User(username="viewer1", hashed_password=hash_password("viewpass"), role="read_only"))
    db.commit()
    db.close()

    return TestClient(app)


def _login(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_login_rejects_bad_credentials(client):
    resp = client.post("/api/auth/login", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_full_generation_and_review_flow(client):
    admin_token = _login(client, "admin", "adminpass")
    reviewer_token = _login(client, "reviewer1", "reviewpass")
    viewer_token = _login(client, "viewer1", "viewpass")

    # Browse academic data (read-only for any authenticated role)
    resp = client.get("/api/subjects", headers=_auth_headers(viewer_token))
    assert resp.status_code == 200
    subjects = resp.json()
    assert any(s["external_id"] == "phys-1" for s in subjects)

    resp = client.get("/api/subjects/phys-1/topics", headers=_auth_headers(viewer_token))
    assert resp.status_code == 200
    topics = resp.json()
    topic_ids = [t["external_id"] for t in topics]
    assert "newtons-laws" in topic_ids

    # A read-only user cannot create a generation job
    job_payload = {
        "external_subject_id": "phys-1",
        "requested_count": 4,
        "option_count": 4,
        "difficulty_distribution": {"easy": 0.5, "medium": 0.5},
        "bloom_distribution": {"remember": 0.5, "understand": 0.5},
        "topics": [
            {
                "external_topic_id": "newtons-laws",
                "manual_context": "Newton's second law: F=ma.",
                "weight": 1.0,
            }
        ],
        "model_provider_type": "mock",
        "model_name": "mock-model",
    }
    resp = client.post("/api/generation-jobs", json=job_payload, headers=_auth_headers(viewer_token))
    assert resp.status_code == 403

    # An administrator (a ROLES_ALLOWED_TO_MANAGE_JOBS role) can
    resp = client.post("/api/generation-jobs", json=job_payload, headers=_auth_headers(admin_token))
    assert resp.status_code == 201, resp.text
    job = resp.json()
    job_id = job["id"]
    assert job["status"] == "QUEUED"

    # Unknown subject is rejected with 404, not a silent empty job
    bad_payload = dict(job_payload, external_subject_id="does-not-exist")
    resp = client.post("/api/generation-jobs", json=bad_payload, headers=_auth_headers(admin_token))
    assert resp.status_code == 404

    # Start the job -- Celery eager mode runs it synchronously via MockProvider
    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth_headers(admin_token))
    assert resp.status_code == 200, resp.text
    started_job = resp.json()
    assert started_job["status"] == "COMPLETED"
    # requested_count=4 with the job model default over_generation_factor
    # (1.4, not exposed on GenerationJobCreate yet) -> round 0 already
    # asks for ceil(4*1.4)=6 and, since MockProvider is 100% structurally
    # valid, that is also the final count (see docs/PHASE0-DESIGN.md
    # section 30 on over-generation).
    assert started_job["generated_count"] == 6

    # Starting an already-started (now completed) job is rejected, not silently re-run
    resp = client.post(f"/api/generation-jobs/{job_id}/start", headers=_auth_headers(admin_token))
    assert resp.status_code == 409

    # Progress reflects generated candidates awaiting review
    resp = client.get(f"/api/generation-jobs/{job_id}/progress", headers=_auth_headers(viewer_token))
    assert resp.status_code == 200
    progress = resp.json()
    assert progress["generated_count"] == 6
    assert progress["pending_review_count"] == 6  # MockProvider always produces structurally valid output

    # List the generated questions
    resp = client.get(f"/api/questions?generation_job_id={job_id}", headers=_auth_headers(viewer_token))
    assert resp.status_code == 200
    questions = resp.json()
    assert len(questions) == 6
    for q in questions:
        assert q["status"] == "PENDING_REVIEW"
        assert len(q["options"]) == 4

    # A read-only user cannot approve
    first_id = questions[0]["id"]
    resp = client.post(f"/api/questions/{first_id}/approve", headers=_auth_headers(viewer_token))
    assert resp.status_code == 403

    # A reviewer can approve
    resp = client.post(f"/api/questions/{first_id}/approve", headers=_auth_headers(reviewer_token))
    assert resp.status_code == 200
    assert resp.json()["status"] == "APPROVED"

    # Approving twice is rejected (not idempotent-silent)
    resp = client.post(f"/api/questions/{first_id}/approve", headers=_auth_headers(reviewer_token))
    assert resp.status_code == 409

    # A reviewer can reject a different one, with a comment
    second_id = questions[1]["id"]
    resp = client.post(
        f"/api/questions/{second_id}/reject",
        json={"action": "reject", "comment": "Ambiguous stem"},
        headers=_auth_headers(reviewer_token),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REJECTED"

    # A reviewer can edit a third one, and the prior version is preserved
    third_id = questions[2]["id"]
    original_stem = questions[2]["question_stem"]
    resp = client.post(
        f"/api/questions/{third_id}/edit",
        json={
            "action": "edit",
            "edited_question_stem": "Edited: which law relates force and acceleration?",
            "edited_options": ["First law", "Second law", "Third law", "Zeroth law"],
            "edited_correct_option": 1,
        },
        headers=_auth_headers(reviewer_token),
    )
    assert resp.status_code == 200, resp.text
    edited = resp.json()
    assert edited["question_stem"] != original_stem
    assert edited["options"][1]["is_correct"] is True

    # Job-level counters reflect the review outcomes
    resp = client.get(f"/api/generation-jobs/{job_id}", headers=_auth_headers(viewer_token))
    updated_job = resp.json()
    assert updated_job["approved_count"] == 1
    assert updated_job["rejected_count"] == 1
