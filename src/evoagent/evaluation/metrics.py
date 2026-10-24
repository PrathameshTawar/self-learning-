"""
Benchmark and experiment metrics calculation engine.
"""

from typing import Any, Dict, List
from evoagent.domain.models import TaskResult


class BenchmarkMetricsCalculator:
    """
    Computes rigorous metrics defined in EvoAgent benchmark specifications:
    - Task Resolution Rate (>= 50% target)
    - Regression-Free Rate (>= 95% target): share of candidate patches that passed their
      targeted test without breaking the regression suite (0.0 when no candidate got that far)
    - Patch Validity Rate (>= 95% target)
    - Retry Recovery Rate
    - Memory Improvement Difference
    """

    @staticmethod
    def calculate_metrics(results: List[TaskResult]) -> Dict[str, Any]:
        """Aggregate task results into standardized performance metrics."""
        total_tasks = len(results)
        if total_tasks == 0:
            return {
                "total_tasks": 0,
                "resolved_tasks": 0,
                "resolution_rate_percent": 0.0,
                "regression_free_rate_percent": 0.0,
                "targeted_pass_candidates": 0,
                "patch_validity_rate_percent": 0.0,
                "retry_recovery_rate_percent": 0.0,
                "average_attempts_per_task": 0.0,
                "average_duration_seconds": 0.0,
            }

        resolved_count = sum(1 for r in results if r.is_successful)
        resolution_rate = (resolved_count / total_tasks) * 100.0

        total_patches_generated = 0
        valid_patches_count = 0
        targeted_pass_count = 0
        regression_passed_count = 0

        retryable_initial_failures = 0
        recovered_on_retry = 0

        total_duration = sum(r.total_duration_seconds for r in results)
        total_attempts = sum(max(1, len(r.attempts)) for r in results)

        for res in results:
            # Check retry recovery
            if len(res.attempts) > 1:
                retryable_initial_failures += 1
                if res.is_successful:
                    recovered_on_retry += 1

            for att in res.attempts:
                if att.patch:
                    total_patches_generated += 1
                    if att.patch.is_valid_format:
                        valid_patches_count += 1
                # Accepted patches always passed regression, so measure over every candidate that
                # fixed the targeted test; otherwise the rate is 100% by construction
                if att.evaluation and att.evaluation.targeted_tests_passed:
                    targeted_pass_count += 1
                    if att.evaluation.regression_tests_passed:
                        regression_passed_count += 1

        patch_validity_rate = (
            (valid_patches_count / total_patches_generated * 100.0)
            if total_patches_generated > 0
            else 0.0
        )
        regression_free_rate = (
            (regression_passed_count / targeted_pass_count * 100.0)
            if targeted_pass_count > 0
            else 0.0
        )
        retry_recovery_rate = (
            (recovered_on_retry / retryable_initial_failures * 100.0)
            if retryable_initial_failures > 0
            else 0.0
        )

        return {
            "total_tasks": total_tasks,
            "resolved_tasks": resolved_count,
            "resolution_rate_percent": round(resolution_rate, 2),
            "regression_free_rate_percent": round(regression_free_rate, 2),
            "targeted_pass_candidates": targeted_pass_count,
            "patch_validity_rate_percent": round(patch_validity_rate, 2),
            "retry_recovery_rate_percent": round(retry_recovery_rate, 2),
            "average_attempts_per_task": round(total_attempts / total_tasks, 2),
            "average_duration_seconds": round(total_duration / total_tasks, 2),
        }
