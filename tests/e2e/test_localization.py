"""
Fault localization: without metadata.target_file hints, the planner must find the defective file
from the failing test's output.
"""

import json
from pathlib import Path
import shutil
import pytest
from evoagent.domain.models import Task
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.orchestration.planner import TaskPlanner
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.search_tools import CodeSearcher
from evoagent.tools.test_tools import TestRunnerTool

TASK_FILES = sorted((Path(__file__).resolve().parents[2] / "benchmarks" / "tasks").glob("*/task.json"))


@pytest.mark.parametrize("task_file", TASK_FILES, ids=lambda f: f.parent.name)
def test_planner_finds_the_defective_file_without_hints(task_file: Path, tmp_path: Path):
    data = json.loads(task_file.read_text(encoding="utf-8"))
    expected = data["metadata"]["target_file"]
    repo = tmp_path / "repo"
    shutil.copytree(task_file.parent / data.get("repo_dir", "code"), repo)

    task = Task(
        task_id=data["task_id"],
        repository_path=repo,
        issue_description=data["issue_description"],
        targeted_test=data["targeted_test"],
        metadata={k: v for k, v in data["metadata"].items() if k != "target_file"},
    )
    pre_test = TestRunnerTool(LocalSubprocessSandbox()).run_pytest(task.targeted_test, cwd=repo, timeout_seconds=30)
    planner = TaskPlanner(ExperienceRetriever(InMemoryMemoryStore()))

    assert planner.locate_target_file(task, pre_test, CodeSearcher(repo)) == expected


def _result(output: str):
    from evoagent.domain.models import TestResult

    return TestResult(command="pytest", passed=False, exit_code=1, stdout=output, stderr="", duration_seconds=0.1)


def _repo_with_two_modules(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "pricing.py").write_text("def total(xs):\n    return sum(xs)\n", encoding="utf-8")
    (repo / "tax.py").write_text("def vat(x):\n    return x * 0.2\n", encoding="utf-8")
    (repo / "tests" / "test_shop.py").write_text(
        "from pricing import total\nfrom tax import vat\n\ndef test_vat():\n    assert vat(10) == 2.5\n", encoding="utf-8"
    )
    return repo


def test_assertion_failure_points_to_the_imported_module_named_in_the_output(tmp_path: Path):
    repo = _repo_with_two_modules(tmp_path)
    task = Task(task_id="t", repository_path=repo, issue_description="", targeted_test="tests/test_shop.py -k test_vat")
    output = (
        "tests/test_shop.py:5: in test_vat\n    assert vat(10) == 2.5\n"
        "E   assert 2.0 == 2.5\nE    +  where 2.0 = vat(10)\n"
    )
    planner = TaskPlanner(ExperienceRetriever(InMemoryMemoryStore()))

    assert planner.locate_target_file(task, _result(output), CodeSearcher(repo)) == "tax.py"


def test_absolute_frames_inside_the_repo_count_and_site_packages_do_not(tmp_path: Path):
    repo = _repo_with_two_modules(tmp_path)
    task = Task(task_id="t", repository_path=repo, issue_description="", targeted_test="tests/test_shop.py")
    output = (
        f'File "{repo / "tests" / "test_shop.py"}", line 5, in test_vat\n'
        f'File "{repo / "pricing.py"}", line 2, in total\n'
        'File "/usr/lib/python3.12/functools.py", line 10, in wrapper\n'
    )
    planner = TaskPlanner(ExperienceRetriever(InMemoryMemoryStore()))

    assert planner.locate_target_file(task, _result(output), CodeSearcher(repo)) == "pricing.py"
