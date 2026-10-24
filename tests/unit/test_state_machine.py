"""
Unit tests for TaskState and TaskStateMachine budget and lifecycle invariants.
"""

from pathlib import Path
from unittest.mock import MagicMock
from evoagent.domain.models import (
    ExecutionBudget,
    Task,
    TestResult,
)
from evoagent.domain.states import TaskState
from evoagent.orchestration.state_machine import TaskStateMachine


def test_task_state_properties():
    assert TaskState.QUEUED.is_terminal is False
    assert TaskState.SUCCEEDED.is_successful is True
    assert TaskState.COMPLETED.is_terminal is True
    assert TaskState.FAILED.is_terminal is True
    assert TaskState.FAILED.is_successful is False
    assert TaskState.INVALID_TASK.is_terminal is True
    assert TaskState.INVALID_TASK.is_successful is False


def test_budget_constraints_defaults():
    budget = ExecutionBudget()
    assert budget.max_attempts == 3
    assert budget.timeout_seconds == 60
    assert budget.max_patch_lines == 150


def test_state_machine_rejects_already_passing_pre_test(tmp_path: Path):
    # Setup a dummy repository directory
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def run(): pass\n", encoding="utf-8")

    task = Task(
        task_id="task_already_passing",
        repository_path=repo_dir,
        issue_description="Bug that is already passing",
        targeted_test="tests/test_dummy.py",
    )

    # Mock evaluator returning passed=True on pre-test verification
    mock_evaluator = MagicMock()
    mock_evaluator.verify_reproduction.return_value = TestResult(
        command="pytest tests/test_dummy.py",
        passed=True,
        exit_code=0,
        stdout="1 passed",
        stderr="",
        duration_seconds=0.1,
    )

    mock_planner = MagicMock()
    mock_repair_agent = MagicMock()
    mock_memory_store = MagicMock()
    mock_patch_tool = MagicMock()
    mock_patch_tool.max_patch_lines = 150
    mock_git_inspector = MagicMock()
    mock_test_runner = MagicMock()

    sm = TaskStateMachine(
        planner=mock_planner,
        repair_agent=mock_repair_agent,
        evaluator=mock_evaluator,
        memory_store=mock_memory_store,
        patch_tool=mock_patch_tool,
        git_inspector=mock_git_inspector,
        test_runner=mock_test_runner,
    )

    result = sm.execute_task(task)

    assert result.final_state == TaskState.INVALID_TASK
    assert result.is_successful is False
    assert result.accepted_patch is None
    # Ensure repair agent was never called for an invalid task
    assert mock_repair_agent.generate_candidate_patch.called is False
