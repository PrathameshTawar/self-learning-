"""
Remote repair worker: takes job ids off the queue, runs the repair loop, and stores the result.
Run one worker per model endpoint slot; scale by adding worker processes or machines.
"""

import json
import logging
from pathlib import Path
import shutil
import socket
import tempfile
import threading
from typing import Any, Dict, Optional, Sequence
import uuid
from evoagent.domain.models import ExecutionBudget, Task, TaskResult
from evoagent.memory.store import IMemoryStore
from evoagent.models.base import IModelBackend
from evoagent.orchestration.factory import build_state_machine, task_from_file
from evoagent.sandbox.docker_runner import DockerSandbox
from evoagent.sandbox.interface import ISandboxRunner
from evoagent.service.job_store import IJobStore
from evoagent.service.jobs import JobRecord, JobStatus
from evoagent.service.queue import Delivery, IJobQueue

log = logging.getLogger("evoagent.worker")


class JobInputError(ValueError):
    """The job cannot run as submitted; retrying it would fail the same way."""


def summarize_result(result: TaskResult) -> Dict[str, Any]:
    """What a caller needs from a run; the full prompts and outputs are in the job's trace."""
    return {
        "task_id": result.task_id,
        "final_state": result.final_state.value,
        "is_successful": result.is_successful,
        "duration_seconds": result.total_duration_seconds,
        "accepted_patch": result.accepted_patch.diff_unified if result.accepted_patch else None,
        "target_file": result.accepted_patch.target_file if result.accepted_patch else None,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "state": a.state.value,
                "hypothesis": (a.hypothesis or "")[:1000],
                "feedback": (
                    (a.evaluation.feedback if a.evaluation else None)
                    or (a.patch.validation_error if a.patch else None)
                ),
            }
            for a in result.attempts
        ],
    }


class RepairWorker:
    def __init__(
        self,
        queue: IJobQueue,
        job_store: IJobStore,
        memory_store: IMemoryStore,
        model_backend: IModelBackend,
        sandbox: ISandboxRunner,
        worker_id: Optional[str] = None,
        max_deliveries: int = 3,
        heartbeat_seconds: float = 60.0,
        benchmark_roots: Sequence[Path] = (Path("benchmarks"),),
        allow_local_sandbox: bool = False,
        trace_archive_dir: Optional[Path] = None,
    ) -> None:
        self.queue = queue
        self.job_store = job_store
        self.memory_store = memory_store
        self.model_backend = model_backend
        self.sandbox = sandbox
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.max_deliveries = max_deliveries
        self.heartbeat_seconds = heartbeat_seconds
        self.benchmark_roots = [Path(r).resolve() for r in benchmark_roots]
        self.allow_local_sandbox = allow_local_sandbox
        self.trace_archive_dir = trace_archive_dir

    def _resolve_benchmark_task(self, relative: str) -> Path:
        for root in self.benchmark_roots:
            candidate = (root / relative).resolve()
            if candidate.is_dir():
                candidate = candidate / "task.json"
            # resolve() follows symlinks, so a link pointing outside the root is rejected too
            if candidate.is_relative_to(root) and candidate.name == "task.json" and candidate.is_file():
                return candidate
        raise JobInputError(f"benchmark task {relative!r} not found under {[str(r) for r in self.benchmark_roots]}")

    def _materialize(self, job: JobRecord, workspace: Path) -> Task:
        request = job.request
        if request.benchmark_task:
            # The task keeps its own id: held-out benchmark tasks are kept out of training data by id,
            # so a caller-chosen name would let one leak into the export
            return task_from_file(self._resolve_benchmark_task(request.benchmark_task))

        if not isinstance(self.sandbox, DockerSandbox) and not self.allow_local_sandbox:
            raise JobInputError(
                "Refusing to run uploaded code outside the docker sandbox. Start the worker with "
                "EVOAGENT_SANDBOX_BACKEND=docker, or set EVOAGENT_WORKER_ALLOW_LOCAL_SANDBOX=true for trusted callers."
            )
        repo = workspace / "repo"
        for rel_path, content in (request.files or {}).items():
            target = repo / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return Task(
            task_id=request.task_id,
            repository_path=repo,
            issue_description=request.issue_description,
            targeted_test=request.targeted_test,
            regression_test=request.regression_test,
            budget=ExecutionBudget(max_attempts=request.max_attempts, timeout_seconds=request.timeout_seconds),
            metadata=request.metadata,
        )

    def _heartbeat_loop(self, delivery: Delivery, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            try:
                if not self.queue.heartbeat(delivery, self.worker_id):
                    log.warning("job %s was reclaimed by another worker", delivery.job_id)
                    return
            except Exception as exc:  # a missed heartbeat is retried on the next tick
                log.warning("heartbeat for job %s failed: %s", delivery.job_id, exc)

    def _run(self, job: JobRecord) -> tuple[JobStatus, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        with tempfile.TemporaryDirectory(prefix=f"evojob_{job.job_id[:8]}_") as tmp:
            workspace = Path(tmp)
            try:
                task = self._materialize(job, workspace)
            except JobInputError as exc:
                return JobStatus.ERROR, None, None, str(exc)
            trace_dir = workspace / "traces"
            sm = build_state_machine(self.memory_store, self.sandbox, self.model_backend, trace_dir=trace_dir)
            result = sm.execute_task(task, persist_episode=job.request.learn)

            trace = None
            trace_files = sorted(trace_dir.glob("*.json")) if trace_dir.exists() else []
            if trace_files:
                trace = json.loads(trace_files[-1].read_text(encoding="utf-8"))
                if self.trace_archive_dir:
                    self.trace_archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(trace_files[-1], self.trace_archive_dir / trace_files[-1].name)
            status = JobStatus.SUCCEEDED if result.is_successful else JobStatus.FAILED
            return status, summarize_result(result), trace, None

    def process(self, delivery: Delivery) -> Optional[JobStatus]:
        """Run one delivery to completion and acknowledge it. Returns the job's final status."""
        if delivery.deliveries > self.max_deliveries:
            # Each earlier delivery ended without a stored result: the job crashes or hangs its worker
            self.job_store.finish(
                delivery.job_id, JobStatus.ERROR,
                error=f"Gave up after {delivery.deliveries - 1} deliveries ended without a result (worker crash or timeout).",
            )
            self.queue.ack(delivery)
            return JobStatus.ERROR

        job = self.job_store.claim(delivery.job_id, self.worker_id)
        if job is None:
            # Finished, cancelled, or unknown: a redelivery of work that needs no run
            self.queue.ack(delivery)
            return None

        log.info("worker %s running job %s (%s, delivery %d)", self.worker_id, job.job_id, job.request.task_id, delivery.deliveries)
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat_loop, args=(delivery, stop), daemon=True)
        beat.start()
        try:
            try:
                status, result, trace, error = self._run(job)
            except Exception as exc:
                log.exception("job %s crashed", job.job_id)
                status, result, trace, error = JobStatus.ERROR, None, None, f"{type(exc).__name__}: {exc}"
            # If storing the result fails the delivery stays unacknowledged and is retried
            self.job_store.finish(job.job_id, status, result=result, trace=trace, error=error)
        finally:
            stop.set()
            beat.join(timeout=5)
        self.queue.ack(delivery)
        log.info("job %s finished: %s", job.job_id, status.value)
        return status

    def run_once(self, block_seconds: float = 5.0) -> Optional[JobStatus]:
        delivery = self.queue.reserve(self.worker_id, block_seconds=block_seconds)
        return self.process(delivery) if delivery else None

    def run_forever(self, stop: Optional[threading.Event] = None, block_seconds: float = 5.0) -> None:
        """Process jobs until `stop` is set; the job in progress always finishes first."""
        stop = stop or threading.Event()
        backoff = 1.0
        while not stop.is_set():
            try:
                self.run_once(block_seconds)
                backoff = 1.0
            except Exception as exc:
                log.error("worker loop error (retrying in %.0fs): %s", backoff, exc)
                stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
