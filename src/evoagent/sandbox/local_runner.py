"""
Local subprocess runner enforcing timeouts, environment sanitization, and output bounds.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from evoagent.sandbox.interface import ExecutionOutcome, ISandboxRunner


# Sensitive environment keys that must never be exposed to candidate code execution
DENIED_ENV_VARS: List[str] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "EVOAGENT_API_KEY",
    "SSH_AUTH_SOCK",
]

MAX_OUTPUT_BYTES = 500_000  # 500 KB output capture limit to avoid OOM


class LocalSubprocessSandbox(ISandboxRunner):
    """
    Subprocess-based sandbox runner with security guardrails:
    - Strips sensitive host secrets
    - Enforces hard execution timeouts
    - Limits standard output and error buffers
    """

    def __init__(self, sanitize_env: bool = True) -> None:
        self.sanitize_env = sanitize_env

    def _build_sanitized_env(self, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct a clean environment without host secrets."""
        env = dict(os.environ)

        if self.sanitize_env:
            for key in DENIED_ENV_VARS:
                env.pop(key, None)

        if extra_env:
            env.update(extra_env)

        return env

    def execute_command(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 60,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> ExecutionOutcome:
        """Execute a command within cwd using subprocess with strict bounds."""
        clean_env = self._build_sanitized_env(extra_env)
        start_time = time.perf_counter()

        try:
            process = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=clean_env,
            )
            duration = time.perf_counter() - start_time

            # Truncate output if larger than limit
            stdout = process.stdout[:MAX_OUTPUT_BYTES]
            stderr = process.stderr[:MAX_OUTPUT_BYTES]

            return ExecutionOutcome(
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=False,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            stdout_str = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
            stderr_str = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
            return ExecutionOutcome(
                exit_code=124,  # Standard timeout exit code
                stdout=stdout_str[:MAX_OUTPUT_BYTES],
                stderr=f"Execution timed out after {timeout_seconds}s.\n{stderr_str[:MAX_OUTPUT_BYTES]}",
                duration_seconds=duration,
                timed_out=True,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            return ExecutionOutcome(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution failed: {str(exc)}",
                duration_seconds=duration,
                timed_out=False,
            )
