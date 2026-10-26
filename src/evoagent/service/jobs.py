"""
Job records for remote repair: what a caller submits and what a worker reports back.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"   # the repair was accepted by the evaluator
    FAILED = "failed"         # the agent ran but no patch was accepted (or the task was invalid)
    ERROR = "error"           # the job could not be run (bad input, worker crash, infrastructure)
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.ERROR, JobStatus.CANCELLED)


def check_relative_path(path: str) -> str:
    """Reject paths that could escape the job workspace once written to disk."""
    pure = PurePosixPath(path.replace("\\", "/"))
    # ":" in the first part catches Windows drive letters such as C:/
    if not pure.parts or pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
        raise ValueError(f"{path!r} must be a relative path inside the repository")
    return str(pure)


class RepairJobRequest(BaseModel):
    """
    A repair task. The repository travels with the job (`files`) so workers need no shared disk;
    `benchmark_task` instead names a task.json that is already on every worker.
    """

    task_id: str = Field(
        ..., min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$",
        description="Caller's label; benchmark_task jobs run under the task file's own id",
    )
    issue_description: str = Field(default="", max_length=20_000)
    targeted_test: str = Field(default="", max_length=500)
    regression_test: Optional[str] = Field(default=None, max_length=500)
    files: Optional[Dict[str, str]] = Field(default=None, description="Repository files: relative path -> content")
    benchmark_task: Optional[str] = Field(default=None, description="task.json path under a worker benchmark root")
    max_attempts: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=60, ge=5, le=600)
    learn: bool = Field(default=True, description="Persist the outcome into shared memory")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_source(self) -> "RepairJobRequest":
        if (self.files is None) == (self.benchmark_task is None):
            raise ValueError("give exactly one of files or benchmark_task")
        if self.files is not None:
            if not self.files:
                raise ValueError("files must not be empty")
            if not self.targeted_test:
                raise ValueError("targeted_test is required with files")
            self.files = {check_relative_path(path): content for path, content in self.files.items()}
        if self.benchmark_task is not None:
            self.benchmark_task = check_relative_path(self.benchmark_task)
        return self

    def payload_bytes(self) -> int:
        return sum(len(path) + len(content.encode("utf-8")) for path, content in (self.files or {}).items())


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    request: RepairJobRequest
    result: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    deliveries: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def summary(self) -> Dict[str, Any]:
        """The record without the uploaded files and trace, for listings and status polling."""
        data = self.model_dump(mode="json", exclude={"trace"})
        data["request"].pop("files", None)
        data["request"]["file_count"] = len(self.request.files or {})
        return data
