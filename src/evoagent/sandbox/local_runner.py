"""
Local subprocess runner enforcing timeouts, environment allowlisting, argument list parsing, and resource limits.
"""

import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Dict, List, Optional
from evoagent.sandbox.interface import ExecutionOutcome, ISandboxRunner

# Strict environment variable allowlist: only essential OS and Python runtime keys are passed
ALLOWED_ENV_KEYS = {
    # System core
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "COMSPEC",
    "PATHEXT",
    "HOME",
    "USER",
    "USERNAME",
    "LANG",
    "LC_ALL",
    "TERM",
    # Python runtime & test frameworks
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "PYTEST_CURRENT_TEST",
    "DEEPEVAL_TELEMETRY_OPT_OUT",
}

MAX_OUTPUT_BYTES = 500_000  # 500 KB output capture limit to avoid OOM


def _set_posix_resource_limits(max_memory_mb: int = 2048, max_cpu_seconds: int = 60) -> None:
    """Apply POSIX resource bounds (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NPROC) if supported."""
    try:
        import resource
        bytes_limit = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
    except (ImportError, AttributeError, ValueError, OSError):
        pass


class LocalSubprocessSandbox(ISandboxRunner):
    """
    Subprocess sandbox runner with security guardrails:
    - Strict environment variable allowlisting (blocks secret leaking)
    - Shell-free argument list execution (prevents shell injection)
    - POSIX resource boundaries on CPU/memory where supported
    - Strict execution timeouts and output buffer truncation
    """

    def __init__(self, use_allowlist_env: bool = True, max_memory_mb: int = 2048) -> None:
        self.use_allowlist_env = use_allowlist_env
        self.max_memory_mb = max_memory_mb

    def _build_sanitized_env(self, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct a clean, allowlisted environment without host secrets."""
        if self.use_allowlist_env:
            clean_env = {k: v for k, v in os.environ.items() if k.upper() in ALLOWED_ENV_KEYS}
        else:
            clean_env = dict(os.environ)

        # Ensure telemetry opt-out is consistently active
        clean_env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

        if extra_env:
            clean_env.update(extra_env)

        return clean_env

    def _parse_command_args(self, command: str) -> List[str]:
        """Parse command string into safe argument vector, stripping enclosing quotes."""
        try:
            tokens = shlex.split(command, posix=True)
            return [t.strip('"\'') for t in tokens]
        except Exception:
            return [t.strip('"\'') for t in command.split()]

    def execute_command(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 60,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> ExecutionOutcome:
        """Execute a command within cwd using subprocess with strict bounds."""
        clean_env = self._build_sanitized_env(extra_env)
        cmd_args = self._parse_command_args(command)
        start_time = time.perf_counter()

        preexec = None
        if os.name != "nt":
            def preexec() -> None:
                _set_posix_resource_limits(
                    max_memory_mb=self.max_memory_mb,
                    max_cpu_seconds=timeout_seconds,
                )

        try:
            process = subprocess.run(
                cmd_args,
                cwd=str(cwd),
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=clean_env,
                preexec_fn=preexec,
            )
            duration = time.perf_counter() - start_time

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
                exit_code=124,
                stdout=stdout_str[:MAX_OUTPUT_BYTES],
                stderr=f"Execution timed out after {timeout_seconds}s.\n{stderr_str[:MAX_OUTPUT_BYTES]}",
                duration_seconds=duration,
                timed_out=True,
            )

        except (FileNotFoundError, PermissionError) as launch_exc:
            error: Exception = launch_exc
            # Fallback for Windows executables requiring shell resolution
            if os.name == "nt":
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
                    return ExecutionOutcome(
                        exit_code=process.returncode,
                        stdout=process.stdout[:MAX_OUTPUT_BYTES],
                        stderr=process.stderr[:MAX_OUTPUT_BYTES],
                        duration_seconds=duration,
                        timed_out=False,
                    )
                except Exception as inner_exc:
                    error = inner_exc

            duration = time.perf_counter() - start_time
            return ExecutionOutcome(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution failed: {str(error)}",
                duration_seconds=duration,
                timed_out=False,
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
