"""
Unit tests for TaskState and TaskStateMachine budget and lifecycle invariants.
"""

from evoagent.domain.models import ExecutionBudget, Task
from evoagent.domain.states import TaskState


def test_task_state_properties():
    assert TaskState.QUEUED.is_terminal is False
    assert TaskState.SUCCEEDED.is_successful is True
    assert TaskState.COMPLETED.is_terminal is True
    assert TaskState.FAILED.is_terminal is True
    assert TaskState.FAILED.is_successful is False


def test_budget_constraints_defaults():
    budget = ExecutionBudget()
    assert budget.max_attempts == 3
    assert budget.timeout_seconds == 60
    assert budget.max_patch_lines == 150
