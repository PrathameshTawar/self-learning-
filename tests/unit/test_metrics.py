"""
Unit tests for BenchmarkMetricsCalculator aggregate metrics.
"""

from evoagent.domain.models import AttemptRecord, EvaluationScore, TaskResult
from evoagent.domain.states import TaskState
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator


def _attempt(number: int, targeted: bool, regression: bool) -> AttemptRecord:
    accepted = targeted and regression
    return AttemptRecord(
        attempt_number=number,
        state=TaskState.SUCCEEDED if accepted else TaskState.RETRYABLE_FAILURE,
        evaluation=EvaluationScore(
            targeted_tests_passed=targeted,
            regression_tests_passed=regression,
            patch_syntactically_valid=True,
            is_accepted=accepted,
        ),
    )


def test_regression_free_rate_counts_candidates_that_broke_other_tests():
    # Attempt 1 fixed the targeted test but broke the regression suite; attempt 2 was clean
    resolved = TaskResult(
        task_id="t1",
        final_state=TaskState.COMPLETED,
        is_successful=True,
        attempts=[_attempt(1, targeted=True, regression=False), _attempt(2, targeted=True, regression=True)],
    )
    metrics = BenchmarkMetricsCalculator.calculate_metrics([resolved])

    assert metrics["targeted_pass_candidates"] == 2
    assert metrics["regression_free_rate_percent"] == 50.0


def test_regression_free_rate_gives_no_credit_when_nothing_was_fixed():
    unresolved = TaskResult(
        task_id="t2",
        final_state=TaskState.FAILED,
        is_successful=False,
        attempts=[_attempt(1, targeted=False, regression=False)],
    )
    metrics = BenchmarkMetricsCalculator.calculate_metrics([unresolved])

    assert metrics["resolution_rate_percent"] == 0.0
    assert metrics["targeted_pass_candidates"] == 0
    assert metrics["regression_free_rate_percent"] == 0.0
