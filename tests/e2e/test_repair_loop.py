"""
End-to-end integration test for the autonomous repair and continuous learning loop.
"""

from pathlib import Path
from evoagent.domain.models import ExecutionBudget, Task
from evoagent.domain.states import TaskState
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.models.heuristic_model import HeuristicCodingModel
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool


def test_autonomous_repair_loop_e2e(temp_repo: Path, local_sandbox: LocalSubprocessSandbox):
    """
    Full end-to-end milestone test:
    1. Reproduce failure on clean repository
    2. Plan and generate patch
    3. Apply patch inside repository
    4. Independently verify fix and regression tests
    5. Save verified episode to memory store
    6. Verify episode retrieval on subsequent task
    """
    memory_store = InMemoryMemoryStore()
    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)
    test_runner = TestRunnerTool(local_sandbox)
    evaluator = IndependentEvaluator(test_runner)
    model = HeuristicCodingModel()
    patch_tool = PatchTool(root_path=temp_repo)
    repair_agent = RepairAgent(model, patch_tool)
    git_inspector = GitInspector(local_sandbox)

    state_machine = TaskStateMachine(
        planner=planner,
        repair_agent=repair_agent,
        evaluator=evaluator,
        memory_store=memory_store,
        patch_tool=patch_tool,
        git_inspector=git_inspector,
        test_runner=test_runner,
    )

    task = Task(
        task_id="e2e-parser-fix",
        repository_path=temp_repo,
        issue_description="first_field crashes with IndexError on empty string",
        targeted_test="tests/test_parser.py -k test_empty_input",
        regression_test="tests/test_parser.py",
        budget=ExecutionBudget(max_attempts=3, timeout_seconds=30),
        metadata={"target_file": "parser.py"},
    )

    result = state_machine.execute_task(task)

    # Invariants
    assert result.is_successful is True
    assert result.final_state == TaskState.COMPLETED
    assert result.accepted_patch is not None
    assert "return parts[0].strip() if parts else" in result.accepted_patch.diff_unified

    # Verify memory persistence
    episodes = memory_store.get_accepted_episodes()
    assert len(episodes) == 1
    assert episodes[0].task_id == "e2e-parser-fix"
    assert episodes[0].outcome == "accepted"

    # Verify that future task retrieval finds this lesson
    lessons = retriever.retrieve_relevant_lessons("IndexError: list index out of range")
    assert len(lessons) >= 1
    assert "e2e-parser-fix" in lessons[0]


def test_autonomous_multi_file_repair_e2e(tmp_path: Path, local_sandbox: LocalSubprocessSandbox):
    """
    Milestone 2 integration test:
    Verify that EvoAgent traverses cross-file AST dependencies to fix a bug in validator.py
    called from service.py.
    """
    repo = tmp_path / "multifile_repo"
    repo.mkdir()

    # Create validator.py with NoneType defect
    (repo / "validator.py").write_text(
        "def normalize_and_validate(raw_text: str) -> str:\n"
        "    cleaned = raw_text.strip()\n"
        "    return cleaned\n",
        encoding="utf-8",
    )

    # Create service.py calling validator
    (repo / "service.py").write_text(
        "from validator import normalize_and_validate\n\n"
        "def process_order(order_id: str, desc: str) -> dict:\n"
        "    cleaned = normalize_and_validate(desc)\n"
        "    return {'id': order_id, 'desc': cleaned}\n",
        encoding="utf-8",
    )

    # Create test suite
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_service.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "from service import process_order\n\n"
        "def test_process_normal():\n"
        "    res = process_order('1', ' item ')\n"
        "    assert res['desc'] == 'item'\n\n"
        "def test_process_none():\n"
        "    res = process_order('2', None)\n"
        "    assert res['desc'] == ''\n",
        encoding="utf-8",
    )

    memory_store = InMemoryMemoryStore()
    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)
    test_runner = TestRunnerTool(local_sandbox)
    evaluator = IndependentEvaluator(test_runner)
    model = HeuristicCodingModel()
    patch_tool = PatchTool(root_path=repo)
    repair_agent = RepairAgent(model, patch_tool)
    git_inspector = GitInspector(local_sandbox)

    state_machine = TaskStateMachine(
        planner=planner,
        repair_agent=repair_agent,
        evaluator=evaluator,
        memory_store=memory_store,
        patch_tool=patch_tool,
        git_inspector=git_inspector,
        test_runner=test_runner,
    )

    task = Task(
        task_id="e2e-multifile-fix",
        repository_path=repo,
        issue_description="service.py crashes with AttributeError when validator.normalize_and_validate receives None",
        targeted_test="tests/test_service.py -k test_process_none",
        regression_test="tests/test_service.py",
        budget=ExecutionBudget(max_attempts=3, timeout_seconds=30),
        metadata={"target_file": "validator.py"},
    )

    result = state_machine.execute_task(task)

    assert result.is_successful is True
    assert result.final_state == TaskState.COMPLETED
    assert result.accepted_patch is not None
    assert result.accepted_patch.target_file == "validator.py"
