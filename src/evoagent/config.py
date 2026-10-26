"""
Configuration management for EvoAgent using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """EvoAgent runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EVOAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    model_provider: Literal["heuristic", "openai_compatible", "ollama", "mock"] = Field(
        default="heuristic",
        description="Inference provider backend",
    )
    model_name: str = Field(
        default="heuristic-v1",
        description="Model identifier or weights name",
    )
    api_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for OpenAI-compatible or Ollama endpoints",
    )
    api_key: str = Field(
        default="",
        description="API key if required by model endpoint",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Seconds to wait for one model response (CPU inference can be slow)",
    )

    # Sandbox
    sandbox_backend: Literal["local", "docker"] = Field(
        default="local",
        description="local: host subprocess; docker: throwaway container with no network",
    )
    sandbox_image: str = Field(
        default="evoagent-sandbox:latest",
        description="Image for the docker sandbox (see docker/sandbox.Dockerfile)",
    )

    # Budget & Execution limits
    timeout_seconds: int = Field(
        default=60,
        description="Maximum seconds allowed for sandbox test execution",
    )
    max_repair_attempts: int = Field(
        default=3,
        description="Maximum retry attempts per task",
    )
    max_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool invocations per attempt",
    )
    max_patch_lines: int = Field(
        default=150,
        description="Maximum lines in a candidate patch diff",
    )

    # Memory settings
    memory_backend: Literal["sqlite", "postgres", "in_memory"] = Field(
        default="sqlite",
        description="Storage engine for episodic memory",
    )
    memory_db_path: Path = Field(
        default=Path("data/evoagent_memory.db"),
        description="File path for SQLite episodic memory",
    )
    database_url: str = Field(
        default="",
        description="PostgreSQL connection string if using postgres backend",
    )

    # Embeddings for pgvector retrieval (postgres memory backend)
    embedding_provider: Literal["hashing", "openai_compatible"] = Field(
        default="hashing",
        description="hashing: dependency-free feature hashing; openai_compatible: an /embeddings endpoint",
    )
    embedding_dim: int = Field(default=384, description="Vector size; fixed when the episodes table is created")
    embedding_model: str = Field(default="nomic-embed-text", description="Model for openai_compatible embeddings")
    embedding_base_url: str = Field(default="", description="Embeddings endpoint (defaults to api_base_url)")

    # Remote workers & API (Milestone 5)
    job_store_backend: Literal["memory", "postgres"] = Field(
        default="memory",
        description="Job records: memory (single process) or postgres (shared by API and workers)",
    )
    queue_backend: Literal["memory", "redis"] = Field(
        default="memory",
        description="Job dispatch: memory (single process) or redis (remote workers)",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    queue_name: str = Field(default="evoagent:jobs", description="Redis stream holding queued job ids")
    queue_visibility_timeout_seconds: int = Field(
        default=300,
        description="A job whose worker sent no heartbeat for this long is handed to another worker",
    )
    queue_max_deliveries: int = Field(
        default=3,
        description="Deliveries before a job that keeps crashing its worker is marked as an error",
    )
    api_tokens: str = Field(default="", description="Comma-separated bearer tokens accepted by the API")
    max_upload_bytes: int = Field(default=2_000_000, description="Largest inline repository a job may carry")
    worker_allow_local_sandbox: bool = Field(
        default=False,
        description="Let workers run uploaded code without the docker sandbox (trusted callers only)",
    )
    worker_benchmark_roots: str = Field(
        default="benchmarks",
        description="Comma-separated directories that benchmark_task job references may point into",
    )

    # Observability & Tracing
    log_level: str = Field(default="INFO", description="Console and trace log level")
    trace_dir: Path = Field(
        default=Path("logs/traces"),
        description="Directory for detailed task traces",
    )

    @staticmethod
    def split_list(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_backend == "sqlite":
            self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
