"""
Unit tests for IndependentEvaluator regression verification and authority enforcement.
"""

from pathlib import Path
from evoagent.domain.models import PatchProposal, Task, TestResult
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.tools.test_tools import TestRunnerTool


class DummyTestRunner(TestRunnerTool):
    """Mock test runner for controlling test outcomes in evaluator tests."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.call_count = 0

    def run_pytest(self, test_path, cwd, timeout_seconds=60, extra_flags=""):
        res = self.outcomes[self.call_count]
        self.call_count += 1
        return res


def test_evaluator_rejects_regression_failure():
    """Verify that a patch passing targeted tests but breaking regressions is rejected."""
    target_result = TestResult(command="pytest targeted", passed=True, exit_code=0, stdout="", stderr="", duration_seconds=0.1)
    dummy_runner = DummyTestRunner([
        TestResult(command="pytest regression", passed=False, exit_code=1, stdout="FAILED test_other", stderr="", duration_seconds=0.1, error_summary="FAILED test_other"),
    ])

    evaluator = IndependentEvaluator(dummy_runner)
    task = Task(
        task_id="task-regression",
        repository_path=Path("."),
        issue_description="Fix bug",
        targeted_test="tests/test_fix.py",
        regression_test="tests/test_all.py",
    )
    patch = PatchProposal(target_file="app.py", diff_unified="--- a/app.py\n+++ b/app.py\n+fixed")

    score = evaluator.evaluate_repair(task, patch, target_result)

    assert score.targeted_tests_passed is True
    assert score.regression_tests_passed is False
    assert score.is_accepted is False
    assert "caused regression" in score.feedback


def test_evaluator_accepts_clean_fix():
    """Verify clean fix passing both targeted and regression suites is accepted."""
    target_result = TestResult(command="pytest targeted", passed=True, exit_code=0, stdout="", stderr="", duration_seconds=0.1)
    dummy_runner = DummyTestRunner([
        TestResult(command="pytest regression", passed=True, exit_code=0, stdout="2 passed", stderr="", duration_seconds=0.1),
    ])

    evaluator = IndependentEvaluator(dummy_runner)
    task = Task(
        task_id="task-clean",
        repository_path=Path("."),
        issue_description="Fix bug",
        targeted_test="tests/test_fix.py",
        regression_test="tests/test_all.py",
    )
    patch = PatchProposal(target_file="app.py", diff_unified="--- a/app.py\n+++ b/app.py\n+fixed")

    score = evaluator.evaluate_repair(task, patch, target_result)

    assert score.targeted_tests_passed is True
    assert score.regression_tests_passed is True
    assert score.is_accepted is True
    assert score.score == 1.0
