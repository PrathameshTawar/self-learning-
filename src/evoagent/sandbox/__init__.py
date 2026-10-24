"""
Sandbox execution interfaces and runners.
"""

from evoagent.sandbox.interface import ISandboxRunner, ExecutionOutcome
from evoagent.sandbox.local_runner import LocalSubprocessSandbox

__all__ = ["ISandboxRunner", "ExecutionOutcome", "LocalSubprocessSandbox"]
