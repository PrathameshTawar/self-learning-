"""
Integration tests against real PostgreSQL (with pgvector) and Redis. Skipped unless
EVOAGENT_TEST_DATABASE_URL and EVOAGENT_TEST_REDIS_URL point at disposable instances; CI starts both.
"""

import os
from pathlib import Path
import time
import uuid
import pytest
from evoagent.memory.embeddings import HashingEmbedder
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.schemas import MemoryEpisodeRecord
from evoagent.service.jobs import JobRecord, JobStatus
from tests.unit.test_service import make_worker, parser_request
from tests.unit.test_state_machine import GOOD_FIX

DATABASE_URL = os.environ.get("EVOAGENT_TEST_DATABASE_URL", "")
REDIS_URL = os.environ.get("EVOAGENT_TEST_REDIS_URL", "")


@pytest.fixture
def db():
    if not DATABASE_URL:
        pytest.skip("EVOAGENT_TEST_DATABASE_URL not set")
    from evoagent.storage.postgres import PostgresDatabase

    database = PostgresDatabase(DATABASE_URL)
    with database.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS memory_episodes, repair_jobs")
    yield database
    database.close()


@pytest.fixture
def redis_queue():
    if not REDIS_URL:
        pytest.skip("EVOAGENT_TEST_REDIS_URL not set")
    from evoagent.service.queue import RedisStreamQueue

    def make(visibility_timeout_seconds: float = 300, stream: str = f"evoagent:test:{uuid.uuid4().hex}"):
        return RedisStreamQueue(REDIS_URL, stream=stream, visibility_timeout_seconds=visibility_timeout_seconds)

    queues = []

    def factory(*args, **kwargs):
        queues.append(make(*args, **kwargs))
        return queues[-1]

    yield factory
    for q in queues:
        q.client.delete(q.stream)


def episode(task_id: str, signature: str, outcome: str = "accepted") -> MemoryEpisodeRecord:
    return MemoryEpisodeRecord(
        episode_id=str(uuid.uuid4()), task_id=task_id, issue_description=f"{task_id} issue",
        failure_signature=signature, target_file="app.py", root_cause_hypothesis="guard missing",
        applied_patch="+    if not items:\n+        return None", outcome=outcome,
        targeted_tests_passed=outcome == "accepted", regression_tests_passed=outcome == "accepted",
        lessons_learned=f"Lesson from {task_id}", metadata={"attempts": 1},
    )


def test_pgvector_memory_shortlists_verified_episodes_for_bm25(db):
    from evoagent.memory.postgres_store import PostgresMemoryStore

    store = PostgresMemoryStore(db, HashingEmbedder(64))
    store.save_episode(episode("empty-list", "IndexError: list index out of range"))
    store.save_episode(episode("none-attr", "AttributeError: 'NoneType' object has no attribute 'get'"))
    store.save_episode(episode("failed-run", "IndexError: list index out of range", outcome="rejected"))

    candidates = store.candidate_episodes("IndexError: list index out of range", limit=1)
    assert [c.task_id for c in candidates] == ["empty-list"]
    assert {e.task_id for e in store.candidate_episodes("anything")} == {"empty-list", "none-attr"}
    assert store.count_episodes() == {"accepted": 2, "rejected": 1}
    assert store.get_all_episodes()[0].metadata == {"attempts": 1}

    lessons = ExperienceRetriever(store).retrieve_relevant_lessons("IndexError: list index out of range")
    assert lessons[0].startswith("[Task empty-list (Resolved)]")
    # A second store with another dimension must refuse the existing table instead of corrupting it
    with pytest.raises(RuntimeError, match="64-dimensional"):
        PostgresMemoryStore(db, HashingEmbedder(32))


def test_postgres_job_store_lifecycle(db, temp_repo: Path):
    from evoagent.service.job_store import PostgresJobStore

    jobs = PostgresJobStore(db)
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    assert jobs.get(job.job_id).request.files == job.request.files

    claimed = jobs.claim(job.job_id, "w1")
    assert (claimed.status, claimed.worker_id, claimed.deliveries) == (JobStatus.RUNNING, "w1", 1)
    assert jobs.cancel(job.job_id) is False
    jobs.finish(job.job_id, JobStatus.FAILED, result={"ok": False}, trace={"task_id": "t", "attempts": []})
    assert jobs.claim(job.job_id, "w2") is None
    assert jobs.list(status=JobStatus.FAILED)[0].trace is None  # listings leave the trace out
    assert jobs.get(job.job_id).trace == {"task_id": "t", "attempts": []}
    assert jobs.traces() == [{"job_id": job.job_id, "task_id": "t", "attempts": []}]
    assert jobs.counts() == {"failed": 1}


def test_redis_queue_reclaims_jobs_from_dead_workers(redis_queue):
    queue = redis_queue(visibility_timeout_seconds=0.3)
    queue.enqueue("job-1")
    first = queue.reserve("dead", block_seconds=0.5)
    assert (first.job_id, first.deliveries) == ("job-1", 1)
    assert queue.reserve("w2", block_seconds=0.1) is None
    assert queue.depth() == {"waiting": 0, "in_flight": 1}

    time.sleep(0.4)
    second = queue.reserve("w2", block_seconds=0.1)
    assert (second.job_id, second.deliveries) == ("job-1", 2)
    assert queue.heartbeat(first, "dead") is False  # must not steal the job back
    assert queue.heartbeat(second, "w2") is True
    queue.ack(second)
    assert queue.depth() == {"waiting": 0, "in_flight": 0}


def test_redis_blocking_read_outlasts_the_client_default_socket_timeout(redis_queue):
    # redis-py 8's default socket timeout is 5 s; an idle worker blocks longer than that
    queue = redis_queue()
    start = time.monotonic()
    assert queue.reserve("w1", block_seconds=6) is None
    assert time.monotonic() - start >= 5.5


def test_redis_heartbeat_prevents_reclaim(redis_queue):
    queue = redis_queue(visibility_timeout_seconds=0.3)
    queue.enqueue("job-1")
    delivery = queue.reserve("w1", block_seconds=0.5)
    for _ in range(4):
        time.sleep(0.15)
        assert queue.heartbeat(delivery, "w1")
    assert queue.reserve("w2", block_seconds=0.1) is None


def test_remote_worker_end_to_end(db, redis_queue, temp_repo: Path):
    from evoagent.memory.postgres_store import PostgresMemoryStore
    from evoagent.service.job_store import PostgresJobStore

    queue, jobs = redis_queue(), PostgresJobStore(db)
    memory = PostgresMemoryStore(db, HashingEmbedder(64))
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    queue.enqueue(job.job_id)

    status = make_worker(queue, jobs, memory, [GOOD_FIX]).run_once(block_seconds=1)

    assert status == JobStatus.SUCCEEDED
    assert jobs.get(job.job_id).result["accepted_patch"] == GOOD_FIX
    assert memory.count_episodes() == {"accepted": 1}
    assert queue.depth() == {"waiting": 0, "in_flight": 0}
