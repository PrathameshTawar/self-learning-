"""
Strategy optimization engine: compares planning policies, prompts, and budgets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
from evoagent.domain.models import TaskResult
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator


@dataclass
class StrategyConfig:
    """Configurable agent hyperparameters and policy settings."""

    strategy_id: str
    prompt_template_style: str  # 'minimal', 'verbose_reasoning', 'memory_dense'
    retrieval_top_k: int = 3
    max_repair_attempts: int = 3
    max_patch_lines: int = 150


class StrategyOptimizer:
    """
    Evaluates alternative planning prompts, retrieval depths, and repair policies (Layer B).
    Selects the winning strategy based on held-out resolution rate.
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, StrategyConfig] = {
            "default": StrategyConfig(
                strategy_id="default",
                prompt_template_style="minimal",
                retrieval_top_k=3,
                max_repair_attempts=3,
            ),
            "deep_memory": StrategyConfig(
                strategy_id="deep_memory",
                prompt_template_style="memory_dense",
                retrieval_top_k=5,
                max_repair_attempts=4,
            ),
        }

    def compare_strategies(
        self,
        results_by_strategy: Dict[str, List[TaskResult]],
    ) -> Dict[str, Any]:
        """Rank strategies based on resolution rate and regression-free rate."""
        rankings = []
        for strat_id, results in results_by_strategy.items():
            metrics = BenchmarkMetricsCalculator.calculate_metrics(results)
            rankings.append({
                "strategy_id": strat_id,
                "resolution_rate": metrics["resolution_rate_percent"],
                "regression_free_rate": metrics["regression_free_rate_percent"],
                "avg_duration": metrics["average_duration_seconds"],
            })

        # Sort descending by resolution rate
        rankings.sort(key=lambda x: (x["resolution_rate"], x["regression_free_rate"]), reverse=True)
        winner = rankings[0]["strategy_id"] if rankings else "default"

        return {
            "winner": winner,
            "rankings": rankings,
        }
