"""
Unit tests for LocalSubprocessSandbox error handling and environment isolation.
"""

from pathlib import Path
import sys
from evoagent.sandbox.local_runner import LocalSubprocessSandbox


def test_missing_executable_returns_failed_outcome(tmp_path: Path):
    sandbox = LocalSubprocessSandbox()
    outcome = sandbox.execute_command("evoagent-no-such-binary --version", cwd=tmp_path)

    assert outcome.exit_code == 1
    assert not outcome.passed
    assert "Sandbox execution failed" in outcome.stderr


def test_host_secrets_are_not_passed_to_child(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    sandbox = LocalSubprocessSandbox()
    outcome = sandbox.execute_command(
        f'"{sys.executable}" -c "import os; print(os.environ.get(\'OPENAI_API_KEY\'))"',
        cwd=tmp_path,
    )

    assert outcome.passed, outcome.stderr
    assert "sk-should-not-leak" not in outcome.stdout
    assert outcome.stdout.strip() == "None"
