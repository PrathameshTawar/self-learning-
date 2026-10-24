"""
Unit tests for genetic strategy representation, operators, and registry.
"""

from pathlib import Path
from evoagent.learning.genetic import (
    GeneticStrategyMutator,
    PromptStyle,
    RetrievalPolicy,
    StrategyChromosome,
)
from evoagent.learning.registry import StrategyRegistry


def test_strategy_chromosome_fitness():
    strat = StrategyChromosome()
    score = strat.compute_fitness(
        resolution_rate=100.0,
        regression_free_rate=100.0,
        avg_duration_seconds=5.0,
        avg_patch_lines=10.0,
    )
    assert score > 0.8
    assert strat.fitness_score == score


def test_genetic_mutator_creates_offspring():
    parent = StrategyChromosome(
        prompt_style=PromptStyle.MINIMAL_DIFF,
        retrieval_top_k=3,
        generation=0,
    )
    child = GeneticStrategyMutator.mutate(parent, mutation_rate=1.0)

    assert child.generation == 1
    assert child.strategy_id != parent.strategy_id


def test_genetic_crossover_combines_parents():
    p1 = StrategyChromosome(prompt_style=PromptStyle.MINIMAL_DIFF, retrieval_top_k=2)
    p2 = StrategyChromosome(prompt_style=PromptStyle.DEFENSIVE_BOUNDARY, retrieval_top_k=5)

    child = GeneticStrategyMutator.crossover(p1, p2)
    assert child.generation == 1
    assert child.retrieval_top_k in (2, 5)


def test_strategy_registry_promotion(tmp_path: Path):
    reg_file = tmp_path / "strategies.json"
    registry = StrategyRegistry(reg_file)

    strat = StrategyChromosome(strategy_id="strat-alpha", prompt_style=PromptStyle.DEFENSIVE_BOUNDARY)
    v_id = registry.promote_strategy(strat, benchmark_evidence={"resolution_rate": 100.0})

    assert v_id == "v1"
    active = registry.get_active_strategy()
    assert active is not None
    assert active.strategy_id == "strat-alpha"
    assert active.prompt_style == PromptStyle.DEFENSIVE_BOUNDARY
