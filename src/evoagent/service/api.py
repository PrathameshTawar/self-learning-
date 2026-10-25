"""
FastAPI job API. Submitting a job only records and enqueues it; workers run it. Callers poll the
job until it reaches a terminal status.
"""

import secrets
from typing import Any, Dict, List, Optional, Sequence
from evoagent.memory.bm25 import BM25Ranker
from evoagent.memory.store import IMemoryStore
from evoagent.service.job_store import IJobStore
from evoagent.service.jobs import JobRecord, JobStatus, RepairJobRequest
from evoagent.service.queue import IJobQueue

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError as exc:  # pragma: no cover - exercised only without the server extra
    raise RuntimeError('The API needs: pip install -e ".[server]"') from exc

# JSON escaping can roughly double the size of source text
BODY_OVERHEAD_FACTOR = 2


def create_app(
    job_store: IJobStore,
    queue: IJobQueue,
    memory_store: Optional[IMemoryStore] = None,
    api_tokens: Sequence[str] = (),
    max_upload_bytes: int = 2_000_000,
    allow_unauthenticated: bool = False,
) -> FastAPI:
    """
    Jobs make workers execute the submitted code's tests, so the API refuses to start without
    bearer tokens unless allow_unauthenticated is set explicitly (local development only).
    """
    tokens = [t for t in api_tokens if t]
    if not tokens and not allow_unauthenticated:
        raise ValueError("Set EVOAGENT_API_TOKENS, or pass --insecure-no-auth for local development")

    app = FastAPI(title="EvoAgent", version="0.1.0", description="Remote repair jobs for the EvoAgent repair loop")
    bearer = HTTPBearer(auto_error=False)
    ranker = BM25Ranker()

    def authorize(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> None:
        if not tokens:
            return
        supplied = credentials.credentials if credentials else ""
        # Compare against every token in constant time so timing does not reveal a prefix match
        if not any(secrets.compare_digest(supplied.encode(), t.encode()) for t in tokens):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: Any) -> Any:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_upload_bytes * BODY_OVERHEAD_FACTOR:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)

    def get_job(job_id: str) -> JobRecord:
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
        return job

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        checks = {"job_store": job_store.ping(), "queue": queue.ping()}
        return JSONResponse({"ready": all(checks.values()), "checks": checks}, status_code=200 if all(checks.values()) else 503)

    @app.post("/v1/jobs", status_code=status.HTTP_201_CREATED, dependencies=[Depends(authorize)])
    def submit_job(request: RepairJobRequest) -> Dict[str, Any]:
        if request.payload_bytes() > max_upload_bytes:
            raise HTTPException(413,
                                f"Repository files exceed {max_upload_bytes} bytes")
        job = JobRecord(request=request)
        # The record is written before the id is queued, so a worker can never receive an unknown job
        job_store.create(job)
        try:
            queue.enqueue(job.job_id)
        except Exception as exc:
            job_store.finish(job.job_id, JobStatus.ERROR, error=f"Could not queue job: {exc}")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job queue unavailable") from exc
        return job.summary()

    @app.get("/v1/jobs", dependencies=[Depends(authorize)])
    def list_jobs(
        status_filter: Optional[JobStatus] = Query(default=None, alias="status"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> List[Dict[str, Any]]:
        return [job.summary() for job in job_store.list(limit=limit, status=status_filter)]

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def job_status(job_id: str) -> Dict[str, Any]:
        return get_job(job_id).summary()

    @app.get("/v1/jobs/{job_id}/trace", dependencies=[Depends(authorize)])
    def job_trace(job_id: str) -> Dict[str, Any]:
        job = get_job(job_id)
        if job.trace is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} has no trace (status: {job.status.value})")
        return job.trace

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_job(job_id: str) -> Dict[str, Any]:
        job = get_job(job_id)
        if not job_store.cancel(job_id):
            raise HTTPException(status.HTTP_409_CONFLICT, f"Only queued jobs can be cancelled (status: {job.status.value})")
        return get_job(job_id).summary()

    @app.get("/v1/stats", dependencies=[Depends(authorize)])
    def stats() -> Dict[str, Any]:
        data: Dict[str, Any] = {"jobs": job_store.counts(), "queue": queue.depth()}
        if memory_store is not None and hasattr(memory_store, "count_episodes"):
            data["memory"] = memory_store.count_episodes()
        return data

    @app.get("/v1/memory/search", dependencies=[Depends(authorize)])
    def search_memory(q: str = Query(..., min_length=1), limit: int = Query(default=5, ge=1, le=50)) -> List[Dict[str, Any]]:
        """Verified lessons the planner would retrieve for this failure text."""
        if memory_store is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No memory store configured")
        episodes = ranker.rank(q, memory_store.candidate_episodes(q), limit=limit)
        return [
            {
                "task_id": ep.task_id,
                "failure_signature": ep.failure_signature,
                "lessons_learned": ep.lessons_learned,
                "applied_patch": ep.applied_patch,
                "created_at": ep.created_at.isoformat(),
            }
            for ep in episodes
        ]

    return app
