"""
Abstract interface for safe code execution sandboxes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecutionOutcome:
    """Standardized result of a sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        """Return True if the execution completed successfully without error."""
        return self.exit_code == 0 and not self.timed_out


class ISandboxRunner(ABC):
    """Protocol for isolating execution of candidate code and test suites."""

    # Python interpreter to invoke inside the sandbox; None means the host's sys.executable
    python_executable: Optional[str] = None

    @abstractmethod
    def execute_command(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 60,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> ExecutionOutcome:
        """
        Execute a shell/python command inside the isolated boundary.
        Enforces timeout limits, memory constraints, and process isolation.
        """
        pass
