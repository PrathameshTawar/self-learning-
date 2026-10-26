"""
Wiring of the repair loop's components, shared by the CLI and remote workers.
"""

import json
from pathlib import Path
from typing import Optional
from evoagent.domain.models import ExecutionBudget, Task
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import IMemoryStore
from evoagent.models.base import IModelBackend
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.interface import ISandboxRunner
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool


def build_state_machine(
    memory_store: IMemoryStore,
    sandbox: ISandboxRunner,
    model_backend: IModelBackend,
    trace_dir: Optional[Path] = None,
) -> TaskStateMachine:
    test_runner = TestRunnerTool(sandbox)
    patch_tool = PatchTool(root_path=Path("."))
    return TaskStateMachine(
        planner=TaskPlanner(ExperienceRetriever(memory_store)),
        repair_agent=RepairAgent(model_backend, patch_tool),
        evaluator=IndependentEvaluator(test_runner),
        memory_store=memory_store,
        patch_tool=patch_tool,
        git_inspector=GitInspector(sandbox),
        test_runner=test_runner,
        trace_dir=trace_dir,
    )


def task_from_file(task_path: Path) -> Task:
    """Build a Task from a benchmark task.json file."""
    data = json.loads(Path(task_path).read_text(encoding="utf-8"))
    return Task(
        task_id=data.get("task_id", task_path.stem),
        repository_path=task_path.parent / data.get("repo_dir", "code"),
        issue_description=data.get("issue_description", ""),
        targeted_test=data.get("targeted_test", ""),
        regression_test=data.get("regression_test"),
        budget=ExecutionBudget(
            max_attempts=data.get("max_attempts", 3),
            timeout_seconds=data.get("timeout_seconds", 60),
        ),
        metadata=data.get("metadata", {}),
    )
