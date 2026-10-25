"""
Build the memory store, job store, queue, and worker described by Settings.
"""

from pathlib import Path
from typing import Optional
from evoagent.config import Settings
from evoagent.memory.embeddings import create_embedder
from evoagent.memory.store import IMemoryStore, InMemoryMemoryStore, SQLiteMemoryStore
from evoagent.models.factory import create_model_backend
from evoagent.sandbox import create_sandbox
from evoagent.service.job_store import IJobStore, InMemoryJobStore, PostgresJobStore
from evoagent.service.queue import IJobQueue, InMemoryJobQueue, RedisStreamQueue
from evoagent.service.worker import RepairWorker
from evoagent.storage.postgres import PostgresDatabase


class Runtime:
    """Holds one shared Postgres pool so the memory and job stores do not each open their own."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._db: Optional[PostgresDatabase] = None

    @property
    def db(self) -> PostgresDatabase:
        if self._db is None:
            self._db = PostgresDatabase(self.settings.database_url)
        return self._db

    def memory_store(self) -> IMemoryStore:
        s = self.settings
        if s.memory_backend == "postgres":
            from evoagent.memory.postgres_store import PostgresMemoryStore

            embedder = create_embedder(
                s.embedding_provider, s.embedding_dim, s.embedding_model,
                s.embedding_base_url or s.api_base_url, s.api_key,
            )
            return PostgresMemoryStore(self.db, embedder)
        if s.memory_backend == "in_memory":
            return InMemoryMemoryStore()
        return SQLiteMemoryStore(s.memory_db_path)

    def job_store(self) -> IJobStore:
        return PostgresJobStore(self.db) if self.settings.job_store_backend == "postgres" else InMemoryJobStore()

    def queue(self) -> IJobQueue:
        s = self.settings
        if s.queue_backend == "redis":
            return RedisStreamQueue(s.redis_url, stream=s.queue_name, visibility_timeout_seconds=s.queue_visibility_timeout_seconds)
        return InMemoryJobQueue(visibility_timeout_seconds=s.queue_visibility_timeout_seconds)

    def worker(
        self, queue: IJobQueue, job_store: IJobStore, memory_store: IMemoryStore, worker_id: Optional[str] = None
    ) -> RepairWorker:
        s = self.settings
        model = create_model_backend(
            s.model_provider, s.model_name, s.api_base_url, s.api_key, timeout=s.request_timeout_seconds
        )
        return RepairWorker(
            queue=queue,
            job_store=job_store,
            memory_store=memory_store,
            model_backend=model,
            sandbox=create_sandbox(s.sandbox_backend, s.sandbox_image),
            worker_id=worker_id,
            max_deliveries=s.queue_max_deliveries,
            # Three heartbeats per visibility window, so one slow heartbeat does not lose the job
            heartbeat_seconds=max(1.0, s.queue_visibility_timeout_seconds / 3),
            benchmark_roots=[Path(r) for r in s.split_list(s.worker_benchmark_roots)],
            allow_local_sandbox=s.worker_allow_local_sandbox,
            trace_archive_dir=s.trace_dir,
        )
