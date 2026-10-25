"""
Targeted test execution tool and output parser.
"""

import sys
from pathlib import Path
from typing import Optional
from evoagent.domain.models import TestResult
from evoagent.sandbox.interface import ISandboxRunner


class TestRunnerTool:
    """Runs test suites inside the execution sandbox and parses test metrics."""

    __test__ = False

    def __init__(self, sandbox: ISandboxRunner, python_executable: Optional[str] = None) -> None:
        self.sandbox = sandbox
        self.python_executable = python_executable or sandbox.python_executable or sys.executable

    def run_pytest(
        self,
        test_path: str,
        cwd: Path,
        timeout_seconds: int = 60,
        extra_flags: str = "-q --tb=short -p no:deepeval",
    ) -> TestResult:
        """Run pytest on a specified test file or node and return parsed TestResult."""
        command = f'"{self.python_executable}" -m pytest {extra_flags} {test_path}'
        outcome = self.sandbox.execute_command(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            extra_env={"DEEPEVAL_TELEMETRY_OPT_OUT": "YES", "PYTHONIOENCODING": "utf-8"},
        )

        error_summary = None
        if not outcome.passed:
            # Extract relevant failure lines
            lines = (outcome.stdout + "\n" + outcome.stderr).splitlines()
            failure_lines = [l for l in lines if "FAILED" in l or "Error" in l or "Exception" in l]
            error_summary = "\n".join(failure_lines[-5:]) if failure_lines else "Test exited with non-zero code"

        return TestResult(
            command=command,
            passed=outcome.passed,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            duration_seconds=outcome.duration_seconds,
            error_summary=error_summary,
        )
