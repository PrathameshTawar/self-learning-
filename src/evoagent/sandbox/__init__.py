"""
Sandbox execution interfaces and runners.
"""

from evoagent.sandbox.interface import ISandboxRunner, ExecutionOutcome
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.sandbox.docker_runner import DockerSandbox


def create_sandbox(backend: str = "local", image: str = "evoagent-sandbox:latest") -> ISandboxRunner:
    """Return the configured sandbox; a Docker sandbox is checked before use so it fails fast."""
    if backend == "docker":
        sandbox = DockerSandbox(image=image)
        sandbox.ensure_ready()
        return sandbox
    return LocalSubprocessSandbox()


__all__ = ["ISandboxRunner", "ExecutionOutcome", "LocalSubprocessSandbox", "DockerSandbox", "create_sandbox"]
