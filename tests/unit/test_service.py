"""
Unit tests for the Milestone 5 service: job validation, queue redelivery, the worker, and the API.
"""

from pathlib import Path
import time
from typing import Dict
import pytest
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.service.job_store import InMemoryJobStore
from evoagent.service.jobs import JobRecord, JobStatus, RepairJobRequest
from evoagent.service.queue import InMemoryJobQueue
from evoagent.service.worker import RepairWorker
from tests.unit.test_state_machine import GOOD_FIX, WRONG_FIX, RecordingModel


def repo_files(repo: Path) -> Dict[str, str]:
    return {p.relative_to(repo).as_posix(): p.read_text(encoding="utf-8") for p in repo.rglob("*.py")}


def parser_request(repo: Path, **overrides) -> RepairJobRequest:
    fields = dict(
        task_id="parser-empty-input",
        issue_description="first_field crashes on empty input",
        targeted_test="tests/test_parser.py -k test_empty_input",
        regression_test="tests/test_parser.py",
        files=repo_files(repo),
        timeout_seconds=30,
    )
    fields.update(overrides)
    return RepairJobRequest(**fields)


def make_worker(queue, job_store, memory_store, diffs, **kwargs) -> RepairWorker:
    kwargs.setdefault("allow_local_sandbox", True)
    return RepairWorker(
        queue=queue,
        job_store=job_store,
        memory_store=memory_store,
        model_backend=RecordingModel(diffs),
        sandbox=LocalSubprocessSandbox(),
        worker_id="w1",
        heartbeat_seconds=0.05,
        **kwargs,
    )


@pytest.mark.parametrize("path", ["../escape.py", "/etc/passwd", "a/../../b.py", "C:/x.py", ""])
def test_job_rejects_paths_outside_the_repository(path: str):
    with pytest.raises(ValueError):
        RepairJobRequest(task_id="t", targeted_test="tests", files={path: "x"})


def test_job_needs_exactly_one_source():
    with pytest.raises(ValueError):
        RepairJobRequest(task_id="t", targeted_test="tests")
    with pytest.raises(ValueError):
        RepairJobRequest(task_id="t", targeted_test="tests", files={"a.py": ""}, benchmark_task="tasks/x/task.json")


def test_unacknowledged_delivery_is_reclaimed_after_the_visibility_timeout():
    queue = InMemoryJobQueue(visibility_timeout_seconds=0.2)
    queue.enqueue("job-1")
    first = queue.reserve("dead-worker", block_seconds=0.1)
    assert first.deliveries == 1
    assert queue.reserve("w2", block_seconds=0.05) is None  # still within the timeout

    time.sleep(0.25)
    second = queue.reserve("w2", block_seconds=0.1)
    assert (second.job_id, second.deliveries) == ("job-1", 2)
    # The original worker has lost the job and must not take it back
    assert queue.heartbeat(first, "dead-worker") is False
    assert queue.heartbeat(second, "w2") is True


def test_heartbeat_keeps_a_long_job_from_being_reclaimed():
    queue = InMemoryJobQueue(visibility_timeout_seconds=0.2)
    queue.enqueue("job-1")
    delivery = queue.reserve("w1", block_seconds=0.1)
    for _ in range(3):
        time.sleep(0.1)
        assert queue.heartbeat(delivery, "w1")
    assert queue.reserve("w2", block_seconds=0.05) is None


def test_worker_repairs_an_uploaded_repository(temp_repo: Path):
    queue, jobs, memory = InMemoryJobQueue(), InMemoryJobStore(), InMemoryMemoryStore()
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    queue.enqueue(job.job_id)

    status = make_worker(queue, jobs, memory, [WRONG_FIX, GOOD_FIX]).run_once(block_seconds=1)

    stored = jobs.get(job.job_id)
    assert status == JobStatus.SUCCEEDED == stored.status
    assert stored.result["accepted_patch"] == GOOD_FIX
    assert [a["state"] for a in stored.result["attempts"]] == ["RETRYABLE_FAILURE", "SUCCEEDED"]
    assert len(stored.trace["attempts"]) == 2
    assert stored.worker_id == "w1" and stored.deliveries == 1
    # learn=True: the verified repair is now shared memory for every worker
    assert len(memory.get_accepted_episodes()) == 1
    assert queue.depth() == {"waiting": 0, "in_flight": 0}


def test_worker_does_not_learn_when_the_job_says_not_to(temp_repo: Path):
    queue, jobs, memory = InMemoryJobQueue(), InMemoryJobStore(), InMemoryMemoryStore()
    job = JobRecord(request=parser_request(temp_repo, learn=False))
    jobs.create(job)
    queue.enqueue(job.job_id)
    make_worker(queue, jobs, memory, [GOOD_FIX]).run_once(block_seconds=1)
    assert memory.get_all_episodes() == []


def test_worker_refuses_uploaded_code_without_the_docker_sandbox(temp_repo: Path):
    queue, jobs = InMemoryJobQueue(), InMemoryJobStore()
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    queue.enqueue(job.job_id)

    worker = make_worker(queue, jobs, InMemoryMemoryStore(), [GOOD_FIX], allow_local_sandbox=False)
    assert worker.run_once(block_seconds=1) == JobStatus.ERROR
    assert "docker sandbox" in jobs.get(job.job_id).error
    assert worker.model_backend.prompts == []


def test_worker_runs_benchmark_tasks_only_from_its_roots(tmp_path: Path):
    benchmarks = Path(__file__).parents[2] / "benchmarks"
    queue, jobs = InMemoryJobQueue(), InMemoryJobStore()
    worker = make_worker(queue, jobs, InMemoryMemoryStore(), [""], benchmark_roots=[benchmarks / "tasks"])

    assert worker._resolve_benchmark_task("bug_001_parser").name == "task.json"
    # A symlink inside the root that points outside it is rejected after resolution
    (tmp_path / "task.json").write_text("{}", encoding="utf-8")
    outside = benchmarks / "tasks" / "zz_link_for_test"
    outside.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(ValueError):
            worker._resolve_benchmark_task("zz_link_for_test")
    finally:
        outside.unlink()


def test_benchmark_jobs_keep_the_benchmark_task_id(tmp_path: Path):
    benchmarks = Path(__file__).parents[2] / "benchmarks"
    queue, jobs = InMemoryJobQueue(), InMemoryJobStore()
    # bug_009 is held out; renaming it would slip its trace past the export's held-out filter
    job = JobRecord(request=RepairJobRequest(task_id="renamed", benchmark_task="tasks/bug_009_latest_event"))
    jobs.create(job)
    queue.enqueue(job.job_id)

    make_worker(queue, jobs, InMemoryMemoryStore(), [""], benchmark_roots=[benchmarks]).run_once(block_seconds=1)

    stored = jobs.get(job.job_id)
    assert stored.result["task_id"] == stored.trace["task_id"] == "bug-009-latest-event"


def test_redelivered_finished_job_is_acknowledged_without_running_again(temp_repo: Path):
    queue, jobs = InMemoryJobQueue(), InMemoryJobStore()
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    jobs.finish(job.job_id, JobStatus.SUCCEEDED, result={"done": True})
    queue.enqueue(job.job_id)

    worker = make_worker(queue, jobs, InMemoryMemoryStore(), [GOOD_FIX])
    assert worker.run_once(block_seconds=1) is None
    assert worker.model_backend.prompts == []
    assert jobs.get(job.job_id).result == {"done": True}
    assert queue.depth()["in_flight"] == 0


def test_job_that_keeps_crashing_its_worker_is_given_up(temp_repo: Path):
    queue, jobs = InMemoryJobQueue(visibility_timeout_seconds=0.01), InMemoryJobStore()
    job = JobRecord(request=parser_request(temp_repo))
    jobs.create(job)
    queue.enqueue(job.job_id)
    for _ in range(3):  # three workers die holding the job
        assert queue.reserve("crashing", block_seconds=0.1)
        time.sleep(0.02)

    worker = make_worker(queue, jobs, InMemoryMemoryStore(), [GOOD_FIX], max_deliveries=3)
    assert worker.run_once(block_seconds=1) == JobStatus.ERROR
    assert "Gave up after 3 deliveries" in jobs.get(job.job_id).error
    assert worker.model_backend.prompts == []
