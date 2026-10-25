"""
Unit tests for LocalSubprocessSandbox error handling and environment isolation.
"""

import os
from pathlib import Path
import sys
import time
import pytest
from evoagent.sandbox.local_runner import LocalSubprocessSandbox


def _is_running(pid: int) -> bool:
    """True while pid exists and is not a zombie waiting to be reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_timeout_kills_processes_spawned_by_the_command(tmp_path: Path):
    # Candidate code (e.g. a test that starts a server) can spawn children; a timeout must not
    # leave them running in the background
    (tmp_path / "spawn.py").write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "open('child.pid', 'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    outcome = LocalSubprocessSandbox().execute_command(
        f'"{sys.executable}" spawn.py', cwd=tmp_path, timeout_seconds=2
    )

    assert outcome.timed_out
    assert outcome.duration_seconds < 10
    child_pid = int((tmp_path / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while _is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _is_running(child_pid)


def test_undecodable_output_does_not_crash(tmp_path: Path):
    outcome = LocalSubprocessSandbox().execute_command(
        f'"{sys.executable}" -c "import sys; sys.stdout.buffer.write(bytes([0xff, 0xfe]))"',
        cwd=tmp_path,
    )
    assert outcome.passed, outcome.stderr
