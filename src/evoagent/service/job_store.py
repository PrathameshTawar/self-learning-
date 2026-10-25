"""
Job store: the source of truth for job status and results. The queue only carries job ids.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import threading
from typing import Any, Dict, List, Optional
from evoagent.service.jobs import JobRecord, JobStatus, RepairJobRequest
from evoagent.storage.postgres import PostgresDatabase


class IJobStore(ABC):
    @abstractmethod
    def create(self, record: JobRecord) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[JobRecord]: ...

    @abstractmethod
    def list(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[JobRecord]: ...

    @abstractmethod
    def claim(self, job_id: str, worker_id: str) -> Optional[JobRecord]:
        """
        Mark a job running on `worker_id` and count the delivery. A job that is already finished or
        cancelled is not claimed (None), so a redelivered message is never run twice.
        """

    @abstractmethod
    def finish(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started; False if it is running or finished."""

    @abstractmethod
    def counts(self) -> Dict[str, int]: ...

    def ping(self) -> bool:
        return True


class InMemoryJobStore(IJobStore):
    """Process-local store for tests and the single-process `serve --embedded-worker` mode."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record.model_copy(deep=True)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[JobRecord]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if status is None or j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def claim(self, job_id: str, worker_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.is_terminal:
                return None
            job.status, job.worker_id = JobStatus.RUNNING, worker_id
            job.deliveries += 1
            job.started_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def finish(self, job_id, status, result=None, trace=None, error=None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.result, job.trace, job.error = status, result, trace, error
            job.finished_at = datetime.now(timezone.utc)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return False
            job.status, job.finished_at = JobStatus.CANCELLED, datetime.now(timezone.utc)
            return True

    def counts(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
            return counts


class PostgresJobStore(IJobStore):
    """Jobs table shared by the API and every worker."""

    def __init__(self, db: PostgresDatabase) -> None:
        self.db = db
        self.db.run_schema("repair_jobs", [
            """
            CREATE TABLE IF NOT EXISTS repair_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request JSONB NOT NULL,
                result JSONB,
                trace JSONB,
                error TEXT,
                worker_id TEXT,
                deliveries INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            )
            """,
            "CREATE INDEX IF NOT EXISTS repair_jobs_status_idx ON repair_jobs (status, created_at DESC)",
        ])

    @staticmethod
    def _to_record(row: Dict[str, Any]) -> JobRecord:
        return JobRecord(**{**row, "request": RepairJobRequest.model_validate(row["request"])})

    def create(self, record: JobRecord) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO repair_jobs (job_id, status, request, deliveries, created_at) VALUES (%s, %s, %s, %s, %s)",
                (record.job_id, record.status.value, record.request.model_dump_json(), record.deliveries,
                 record.created_at),
            )

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM repair_jobs WHERE job_id = %s", (job_id,)).fetchone()
        return self._to_record(row) if row else None

    def list(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[JobRecord]:
        # Listings never need the trace, which holds every prompt of every attempt
        columns = "job_id, status, request, result, NULL AS trace, error, worker_id, deliveries, created_at, started_at, finished_at"
        with self.db.connection() as conn:
            if status is None:
                rows = conn.execute(
                    f"SELECT {columns} FROM repair_jobs ORDER BY created_at DESC LIMIT %s", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {columns} FROM repair_jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status.value, limit),
                ).fetchall()
        return [self._to_record(r) for r in rows]

    def claim(self, job_id: str, worker_id: str) -> Optional[JobRecord]:
        with self.db.connection() as conn:
            row = conn.execute(
                """
                UPDATE repair_jobs SET status = 'running', worker_id = %s, deliveries = deliveries + 1, started_at = now()
                WHERE job_id = %s AND status IN ('queued', 'running')
                RETURNING *
                """,
                (worker_id, job_id),
            ).fetchone()
        return self._to_record(row) if row else None

    def finish(self, job_id, status, result=None, trace=None, error=None) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE repair_jobs SET status = %s, result = %s, trace = %s, error = %s, finished_at = now() "
                "WHERE job_id = %s",
                (
                    status.value,
                    json.dumps(result) if result is not None else None,
                    json.dumps(trace) if trace is not None else None,
                    error,
                    job_id,
                ),
            )

    def cancel(self, job_id: str) -> bool:
        with self.db.connection() as conn:
            row = conn.execute(
                "UPDATE repair_jobs SET status = 'cancelled', finished_at = now() "
                "WHERE job_id = %s AND status = 'queued' RETURNING job_id",
                (job_id,),
            ).fetchone()
        return row is not None

    def counts(self) -> Dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT status, count(*) AS n FROM repair_jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def traces(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Run traces of finished jobs, newest first, for training-data export."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT job_id, trace FROM repair_jobs WHERE trace IS NOT NULL ORDER BY finished_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [{"job_id": r["job_id"], **r["trace"]} for r in rows]

    def ping(self) -> bool:
        return self.db.ping()
