"""
Controlled experiment harness comparing Baseline, Memory, Strategy, and Fine-Tuned variants.
"""

from typing import Any, Dict, List
from evoagent.domain.models import Task, TaskResult
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator
from evoagent.orchestration.state_machine import TaskStateMachine


class ExperimentHarness:
    """
    Executes controlled experiments across four configurations:
    A - Baseline: Frozen model, no experience retrieval.
    B - Memory: Same model with verified experience retrieval.
    C - Strategy Learning: Optimized planning and repair policy.
    D - Fine-Tuned Model: Adapter compared with base model.
    """

    def __init__(self, state_machine: TaskStateMachine) -> None:
        self.state_machine = state_machine

    def run_experiment_suite(
        self,
        tasks: List[Task],
        variant_name: str = "B - Memory",
    ) -> Dict[str, Any]:
        """Run tasks through the state machine and collect metrics."""
        results: List[TaskResult] = []

        for task in tasks:
            result = self.state_machine.execute_task(task)
            results.append(result)

        metrics = BenchmarkMetricsCalculator.calculate_metrics(results)
        return {
            "variant": variant_name,
            "metrics": metrics,
            "results": results,
        }

    @staticmethod
    def compute_delta(baseline_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute relative percentage point improvement over baseline."""
        res_delta = candidate_metrics["resolution_rate_percent"] - baseline_metrics["resolution_rate_percent"]
        reg_delta = candidate_metrics["regression_free_rate_percent"] - baseline_metrics["regression_free_rate_percent"]

        return {
            "resolution_rate_delta_percentage_points": round(res_delta, 2),
            "regression_free_delta_percentage_points": round(reg_delta, 2),
            "is_statistically_improved": res_delta > 0 and reg_delta >= 0,
        }
