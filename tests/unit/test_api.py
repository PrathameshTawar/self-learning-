"""
Unit tests for the FastAPI job API (Milestone 5), driven with an in-process worker.
"""

from pathlib import Path
import pytest
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.service.job_store import InMemoryJobStore
from evoagent.service.jobs import JobStatus
from evoagent.service.queue import InMemoryJobQueue
from tests.unit.test_service import make_worker, parser_request
from tests.unit.test_state_machine import GOOD_FIX


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from evoagent.service.api import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def service():
    queue, jobs, memory = InMemoryJobQueue(), InMemoryJobStore(), InMemoryMemoryStore()
    client = TestClient(create_app(jobs, queue, memory, api_tokens=[TOKEN], max_upload_bytes=50_000))
    return client, queue, jobs, memory


def test_api_refuses_to_start_without_tokens():
    with pytest.raises(ValueError):
        create_app(InMemoryJobStore(), InMemoryJobQueue())


def test_api_requires_a_valid_bearer_token(service, temp_repo: Path):
    client, *_ = service
    body = parser_request(temp_repo).model_dump(mode="json")
    assert client.post("/v1/jobs", json=body).status_code == 401
    assert client.post("/v1/jobs", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/healthz").status_code == 200  # probes stay unauthenticated


def test_api_job_lifecycle_with_a_worker(service, temp_repo: Path):
    client, queue, jobs, memory = service
    res = client.post("/v1/jobs", json=parser_request(temp_repo).model_dump(mode="json"), headers=AUTH)
    assert res.status_code == 201
    job = res.json()
    assert job["status"] == "queued" and "files" not in job["request"] and job["request"]["file_count"] == 2
    assert client.get(f"/v1/jobs/{job['job_id']}/trace", headers=AUTH).status_code == 404

    make_worker(queue, jobs, memory, [GOOD_FIX]).run_once(block_seconds=1)

    done = client.get(f"/v1/jobs/{job['job_id']}", headers=AUTH).json()
    assert done["status"] == "succeeded" and done["result"]["accepted_patch"] == GOOD_FIX
    assert client.get(f"/v1/jobs/{job['job_id']}/trace", headers=AUTH).json()["final_state"] == "COMPLETED"
    assert [j["job_id"] for j in client.get("/v1/jobs?status=succeeded", headers=AUTH).json()] == [job["job_id"]]
    assert client.get("/v1/stats", headers=AUTH).json()["jobs"] == {"succeeded": 1}
    lessons = client.get("/v1/memory/search", params={"q": "IndexError list index out of range"}, headers=AUTH).json()
    assert lessons and lessons[0]["task_id"] == "parser-empty-input"


def test_api_cancels_only_queued_jobs(service, temp_repo: Path):
    client, queue, jobs, memory = service
    job_id = client.post("/v1/jobs", json=parser_request(temp_repo).model_dump(mode="json"), headers=AUTH).json()["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).json()["status"] == "cancelled"
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).status_code == 409
    # The worker skips the cancelled job
    worker = make_worker(queue, jobs, memory, [GOOD_FIX])
    assert worker.run_once(block_seconds=1) is None and worker.model_backend.prompts == []


def test_api_rejects_oversized_uploads(service):
    client, *_ = service
    body = {"task_id": "big", "targeted_test": "tests", "files": {"big.py": "x" * 60_000}}
    assert client.post("/v1/jobs", json=body, headers=AUTH).status_code == 413
    huge = {"task_id": "huge", "targeted_test": "tests", "files": {"big.py": "x" * 200_000}}
    assert client.post("/v1/jobs", json=huge, headers=AUTH).status_code == 413
    assert client.get("/v1/jobs/unknown", headers=AUTH).status_code == 404


def test_api_marks_a_job_as_error_when_the_queue_is_down(service, temp_repo: Path):
    client, queue, jobs, _ = service

    def broken(job_id: str) -> None:
        raise ConnectionError("redis down")

    queue.enqueue = broken
    res = client.post("/v1/jobs", json=parser_request(temp_repo).model_dump(mode="json"), headers=AUTH)
    assert res.status_code == 503
    assert jobs.list()[0].status == JobStatus.ERROR
