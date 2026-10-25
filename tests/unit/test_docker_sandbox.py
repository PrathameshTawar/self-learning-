"""
DockerSandbox tests. The container tests need a Docker daemon and the sandbox image; they are
skipped otherwise. Set EVOAGENT_TEST_SANDBOX_IMAGE to use an image other than evoagent-sandbox:latest.
"""

import os
from pathlib import Path
import subprocess
import pytest
from evoagent.sandbox.docker_runner import DockerSandbox
from evoagent.tools.test_tools import TestRunnerTool

IMAGE = os.environ.get("EVOAGENT_TEST_SANDBOX_IMAGE", "evoagent-sandbox:latest")


def _image_available() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_docker = pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} not available")


def test_container_is_isolated_by_default(tmp_path: Path):
    args = DockerSandbox(image="img").docker_args("evoagent-x", tmp_path, {"A": "1"})

    assert args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert f"{tmp_path.resolve()}:/workspace" in args
    assert args[-1] == "img"
    assert "A=1" in args


def test_test_runner_uses_the_container_interpreter():
    assert TestRunnerTool(DockerSandbox()).python_executable == "python"


def test_missing_image_fails_fast_with_build_instructions():
    with pytest.raises(RuntimeError, match="docker build"):
        DockerSandbox(image="evoagent-no-such-image:never").ensure_ready()


@needs_docker
def test_runs_pytest_inside_the_container(tmp_path: Path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert 1 + 1 == 3\n", encoding="utf-8")
    runner = TestRunnerTool(DockerSandbox(image=IMAGE))

    assert runner.run_pytest("test_ok.py", cwd=tmp_path, timeout_seconds=60).passed
    failed = runner.run_pytest("test_bad.py", cwd=tmp_path, timeout_seconds=60)
    assert failed.exit_code == 1
    assert "FAILED test_bad.py::test_bad" in failed.stdout


@needs_docker
def test_container_has_no_network_and_cannot_see_host_files(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "host_secret.txt"
    secret.write_text("do not read", encoding="utf-8")
    sandbox = DockerSandbox(image=IMAGE)

    net = sandbox.execute_command(
        "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\"", cwd=workspace
    )
    assert net.exit_code != 0
    assert "OSError" in net.stderr or "unreachable" in net.stderr.lower()

    peek = sandbox.execute_command(f"python -c \"print(open('{secret}').read())\"", cwd=workspace)
    assert peek.exit_code != 0
    assert "do not read" not in peek.stdout


@needs_docker
def test_timeout_kills_and_removes_the_container(tmp_path: Path):
    outcome = DockerSandbox(image=IMAGE).execute_command(
        "python -c \"import time; time.sleep(60)\"", cwd=tmp_path, timeout_seconds=3
    )

    assert outcome.timed_out
    assert outcome.duration_seconds < 30
    leftover = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", "name=evoagent-"], capture_output=True, text=True
    )
    assert leftover.stdout.strip() == ""


@needs_docker
def test_full_repair_loop_runs_in_the_container(temp_repo: Path):
    from evoagent.domain.states import TaskState
    from evoagent.evaluation.evaluator import IndependentEvaluator
    from evoagent.memory.retrieval import ExperienceRetriever
    from evoagent.memory.store import InMemoryMemoryStore
    from evoagent.orchestration.planner import TaskPlanner
    from evoagent.orchestration.repair_agent import RepairAgent
    from evoagent.orchestration.state_machine import TaskStateMachine
    from evoagent.tools.git_tools import GitInspector
    from evoagent.tools.patch_tools import PatchTool
    from tests.unit.test_state_machine import GOOD_FIX, RecordingModel, _parser_task

    sandbox = DockerSandbox(image=IMAGE)
    test_runner = TestRunnerTool(sandbox)
    store = InMemoryMemoryStore()
    patch_tool = PatchTool(root_path=temp_repo)
    sm = TaskStateMachine(
        planner=TaskPlanner(ExperienceRetriever(store)),
        repair_agent=RepairAgent(RecordingModel([GOOD_FIX]), patch_tool),
        evaluator=IndependentEvaluator(test_runner),
        memory_store=store,
        patch_tool=patch_tool,
        git_inspector=GitInspector(sandbox),
        test_runner=test_runner,
    )

    result = sm.execute_task(_parser_task(temp_repo))

    assert result.final_state == TaskState.COMPLETED
    assert result.attempts[0].post_test_result.command.startswith('"python" -m pytest')
