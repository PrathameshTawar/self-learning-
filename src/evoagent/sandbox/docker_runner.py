"""
Container sandbox: runs each command in a throwaway Docker container without network access.
"""

import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid
from typing import Dict, List, Optional
from evoagent.sandbox.interface import ExecutionOutcome, ISandboxRunner
from evoagent.sandbox.local_runner import MAX_OUTPUT_BYTES


class DockerSandbox(ISandboxRunner):
    """
    Runs every command in a fresh container from `image`:
    - no network (--network none), read-only root filesystem, private /tmp
    - only the working directory is mounted (at /workspace), so candidate code cannot read or
      modify anything else on the host
    - memory, CPU and process-count limits; on timeout the container is killed and removed
    Build the default image with: docker build -t evoagent-sandbox -f docker/sandbox.Dockerfile .
    """

    python_executable = "python"

    def __init__(
        self,
        image: str = "evoagent-sandbox:latest",
        memory_mb: int = 2048,
        cpus: float = 2.0,
        pids_limit: int = 256,
        docker_binary: str = "docker",
    ) -> None:
        self.image = image
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker_binary = docker_binary

    def ensure_ready(self) -> None:
        """Raise RuntimeError with build instructions if the daemon or the image is unavailable."""
        try:
            probe = subprocess.run(
                [self.docker_binary, "image", "inspect", self.image], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Docker is not available: {exc}") from exc
        if probe.returncode != 0:
            raise RuntimeError(
                f"Sandbox image {self.image!r} not found ({probe.stderr.strip()}). Build it with: "
                "docker build -t evoagent-sandbox -f docker/sandbox.Dockerfile ."
            )

    def docker_args(self, name: str, cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> List[str]:
        """The `docker run` prefix for one command, up to and including the image name."""
        args = [
            self.docker_binary, "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp",
            "--memory", f"{self.memory_mb}m", "--cpus", str(self.cpus), "--pids-limit", str(self.pids_limit),
            "--volume", f"{Path(cwd).resolve()}:/workspace", "--workdir", "/workspace",
        ]
        if hasattr(os, "getuid"):
            # Files written into the mounted workspace stay owned by the calling user
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for key, value in (extra_env or {}).items():
            args += ["--env", f"{key}={value}"]
        return args + [self.image]

    def execute_command(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 60,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> ExecutionOutcome:
        name = f"evoagent-{uuid.uuid4().hex[:12]}"
        args = self.docker_args(name, cwd, extra_env) + shlex.split(command)
        start_time = time.perf_counter()
        try:
            proc = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            # Killing the docker client leaves the container running; remove it explicitly
            subprocess.run([self.docker_binary, "rm", "--force", name], capture_output=True)
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ExecutionOutcome(
                exit_code=124,
                stdout=stdout[:MAX_OUTPUT_BYTES],
                stderr=f"Execution timed out after {timeout_seconds}s.\n{stderr[:MAX_OUTPUT_BYTES]}",
                duration_seconds=time.perf_counter() - start_time,
                timed_out=True,
            )
        except OSError as exc:
            return ExecutionOutcome(
                exit_code=1,
                stdout="",
                stderr=f"Sandbox execution failed: {exc}",
                duration_seconds=time.perf_counter() - start_time,
                timed_out=False,
            )
        return ExecutionOutcome(
            exit_code=proc.returncode,
            stdout=proc.stdout[:MAX_OUTPUT_BYTES],
            stderr=proc.stderr[:MAX_OUTPUT_BYTES],
            duration_seconds=time.perf_counter() - start_time,
            timed_out=False,
        )
