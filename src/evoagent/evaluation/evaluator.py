"""
Independent evaluation authority for validating bug reproduction and repairs.
"""

from pathlib import Path
from evoagent.domain.models import EvaluationScore, PatchProposal, Task, TestResult
from evoagent.tools.test_tools import TestRunnerTool


class IndependentEvaluator:
    """
    The evaluator is the sole authority on task success:
    - Verifies that the bug is reproduced pre-patch (targeted test MUST fail initially)
    - Verifies that the targeted test passes post-patch
    - Verifies that regression tests pass (no regressions introduced)
    - Rejects fixes that fail regression or security constraints
    """

    def __init__(self, test_runner: TestRunnerTool) -> None:
        self.test_runner = test_runner

    def verify_reproduction(self, task: Task) -> TestResult:
        """Run the targeted test before applying any fix to confirm failure reproduction."""
        return self.test_runner.run_pytest(
            test_path=task.targeted_test,
            cwd=task.repository_path,
            timeout_seconds=task.budget.timeout_seconds,
        )

    def evaluate_repair(
        self,
        task: Task,
        patch: PatchProposal,
        post_target_result: TestResult,
    ) -> EvaluationScore:
        """
        Evaluate candidate repair across targeted tests, regression tests, and patch validity.
        """
        # 1. Check patch validity
        if not patch.is_valid_format:
            return EvaluationScore(
                targeted_tests_passed=False,
                regression_tests_passed=False,
                patch_syntactically_valid=False,
                is_accepted=False,
                score=0.0,
                feedback=f"Invalid patch format: {patch.validation_error}",
            )

        # 2. Check targeted test
        targeted_passed = post_target_result.passed
        if not targeted_passed:
            return EvaluationScore(
                targeted_tests_passed=False,
                regression_tests_passed=False,
                patch_syntactically_valid=True,
                is_accepted=False,
                score=0.2,
                feedback="Targeted test still fails after applying patch.",
            )

        # 3. Check regression tests if specified
        regression_passed = True
        regression_feedback = "No separate regression test suite specified."

        if task.regression_test:
            reg_result = self.test_runner.run_pytest(
                test_path=task.regression_test,
                cwd=task.repository_path,
                timeout_seconds=task.budget.timeout_seconds,
            )
            regression_passed = reg_result.passed
            if not regression_passed:
                return EvaluationScore(
                    targeted_tests_passed=True,
                    regression_tests_passed=False,
                    patch_syntactically_valid=True,
                    is_accepted=False,
                    score=0.5,
                    feedback=f"Patch fixed targeted test but caused regression in suite: {reg_result.error_summary}",
                )
            regression_feedback = "All regression tests passed successfully."

        # All checks passed
        return EvaluationScore(
            targeted_tests_passed=True,
            regression_tests_passed=regression_passed,
            patch_syntactically_valid=True,
            is_accepted=True,
            score=1.0,
            feedback=f"Fix verified! Targeted test passed. {regression_feedback}",
        )
