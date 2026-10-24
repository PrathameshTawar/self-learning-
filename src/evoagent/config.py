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

    # Observability & Tracing
    log_level: str = Field(default="INFO", description="Console and trace log level")
    trace_dir: Path = Field(
        default=Path("logs/traces"),
        description="Directory for detailed task traces",
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_backend == "sqlite":
            self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
