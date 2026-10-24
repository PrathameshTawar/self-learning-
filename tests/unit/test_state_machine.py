"""
Unit tests for TaskState and TaskStateMachine budget and lifecycle invariants.
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock
from evoagent.domain.models import (
    ExecutionBudget,
    Task,
    TestResult,
)
from evoagent.domain.states import TaskState
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.learning.genetic import PromptStyle, StrategyChromosome
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool


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


class RecordingModel(IModelBackend):
    """Model backend that replays scripted diffs and records every prompt it receives."""

    def __init__(self, diffs: List[str]) -> None:
        self.diffs = diffs
        self.prompts: List[ModelPrompt] = []

    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        self.prompts.append(prompt)
        diff = self.diffs[min(len(self.prompts), len(self.diffs)) - 1]
        return ModelResponse(hypothesis="scripted", proposed_diff=diff)


GOOD_FIX = (
    "<<<<<<< SEARCH\n"
    "    return parts[0].strip()\n"
    "=======\n"
    "    return parts[0].strip() if parts else ''\n"
    ">>>>>>> REPLACE"
)
WRONG_FIX = GOOD_FIX.replace("else ''", "else None")


def _real_state_machine(repo: Path, model: IModelBackend, store: InMemoryMemoryStore) -> TaskStateMachine:
    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    patch_tool = PatchTool(root_path=repo)
    return TaskStateMachine(
        planner=TaskPlanner(ExperienceRetriever(store)),
        repair_agent=RepairAgent(model, patch_tool),
        evaluator=IndependentEvaluator(test_runner),
        memory_store=store,
        patch_tool=patch_tool,
        git_inspector=GitInspector(sandbox),
        test_runner=test_runner,
    )


def _parser_task(repo: Path, targeted_test: str = "tests/test_parser.py -k test_empty_input", attempts: int = 3) -> Task:
    return Task(
        task_id="parser-empty-input",
        repository_path=repo,
        issue_description="first_field crashes on empty input",
        targeted_test=targeted_test,
        regression_test="tests/test_parser.py",
        budget=ExecutionBudget(max_attempts=attempts, timeout_seconds=30),
        metadata={"target_file": "parser.py"},
    )


def test_strategy_genes_reach_the_model_without_crashing(temp_repo: Path):
    model = RecordingModel([GOOD_FIX])
    store = InMemoryMemoryStore()
    sm = _real_state_machine(temp_repo, model, store)
    strategy = StrategyChromosome(prompt_style=PromptStyle.CHAIN_OF_THOUGHT, temperature=0.7)

    result = sm.execute_task(_parser_task(temp_repo), strategy=strategy, persist_episode=False)

    assert result.final_state == TaskState.COMPLETED
    assert model.prompts[0].temperature == 0.7
    assert "Chain-of-Thought" in model.prompts[0].system_instruction
    # Frozen-memory evaluation must not write episodes
    assert store.get_all_episodes() == []
    # The original repository is never modified
    assert "if parts else" not in (temp_repo / "parser.py").read_text(encoding="utf-8")


def test_strategy_attempts_never_exceed_task_budget(temp_repo: Path):
    store = InMemoryMemoryStore()

    model = RecordingModel([WRONG_FIX])
    result = _real_state_machine(temp_repo, model, store).execute_task(
        _parser_task(temp_repo, attempts=2), strategy=StrategyChromosome(max_repair_attempts=6)
    )
    assert len(result.attempts) == 2

    model = RecordingModel([WRONG_FIX])
    result = _real_state_machine(temp_repo, model, store).execute_task(
        _parser_task(temp_repo, attempts=3), strategy=StrategyChromosome(max_repair_attempts=1)
    )
    assert len(result.attempts) == 1


def test_retry_feedback_contains_post_patch_test_failure(temp_repo: Path):
    model = RecordingModel([WRONG_FIX, GOOD_FIX])
    result = _real_state_machine(temp_repo, model, InMemoryMemoryStore()).execute_task(_parser_task(temp_repo))

    assert result.is_successful
    assert len(model.prompts) == 2
    feedback = model.prompts[1].previous_attempt_error
    assert "Targeted test still fails" in feedback
    assert "AssertionError" in feedback


def test_unrunnable_targeted_test_is_invalid_task(temp_repo: Path):
    model = RecordingModel([GOOD_FIX])
    store = InMemoryMemoryStore()
    task = _parser_task(temp_repo, targeted_test="tests/test_parser.py -k no_such_test")

    result = _real_state_machine(temp_repo, model, store).execute_task(task)

    assert result.final_state == TaskState.INVALID_TASK
    assert "exit code 5" in result.attempts[0].hypothesis
    assert model.prompts == []
    assert store.get_all_episodes() == []
