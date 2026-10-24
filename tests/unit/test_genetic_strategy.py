"""
Unit tests for genetic strategy representation, operators, and registry.
"""

from pathlib import Path
import pytest
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
        max_patch_lines=150,
        temperature=0.2,
        generation=0,
    )
    child = GeneticStrategyMutator.mutate(parent, mutation_rate=1.0)

    assert child.generation == 1
    assert child.strategy_id != parent.strategy_id


def test_genetic_crossover_combines_parents():
    p1 = StrategyChromosome(
        prompt_style=PromptStyle.MINIMAL_DIFF,
        retrieval_top_k=2,
        max_patch_lines=100,
        temperature=0.1,
    )
    p2 = StrategyChromosome(
        prompt_style=PromptStyle.DEFENSIVE_BOUNDARY,
        retrieval_top_k=5,
        max_patch_lines=200,
        temperature=0.7,
    )

    child = GeneticStrategyMutator.crossover(p1, p2)
    assert child.generation == 1
    assert child.retrieval_top_k in (2, 5)
    assert child.max_patch_lines in (100, 200)
    assert child.temperature in (0.1, 0.7)


def test_strategy_registry_promotion(tmp_path: Path):
    reg_file = tmp_path / "strategies.json"
    registry = StrategyRegistry(reg_file)

    strat = StrategyChromosome(
        strategy_id="strat-alpha",
        prompt_style=PromptStyle.DEFENSIVE_BOUNDARY,
        fitness_score=0.75,
    )
    v_id = registry.promote_strategy(strat, benchmark_evidence={"resolution_rate": 100.0})

    assert v_id == "v1"
    active = registry.get_active_strategy()
    assert active is not None
    assert active.strategy_id == "strat-alpha"
    assert active.prompt_style == PromptStyle.DEFENSIVE_BOUNDARY


def test_strategy_registry_statistical_gate_rejects_inferior(tmp_path: Path):
    reg_file = tmp_path / "strategies.json"
    registry = StrategyRegistry(reg_file)

    # 1. Base promotion
    strat_base = StrategyChromosome(strategy_id="strat-base", fitness_score=0.80)
    registry.promote_strategy(strat_base, benchmark_evidence={"fitness": 0.80})

    # 2. Attempt promoting candidate without sufficient improvement (+0.01 < 0.05 min margin)
    strat_weak = StrategyChromosome(strategy_id="strat-weak", fitness_score=0.81)
    with pytest.raises(ValueError, match="did not exceed current baseline"):
        registry.promote_strategy(strat_weak, benchmark_evidence={"fitness": 0.81}, min_fitness_margin=0.05)

    # 3. Promote candidate with statistically meaningful improvement (+0.10 >= 0.05)
    strat_strong = StrategyChromosome(strategy_id="strat-strong", fitness_score=0.90)
    v2 = registry.promote_strategy(strat_strong, benchmark_evidence={"fitness": 0.90}, min_fitness_margin=0.05)
    assert v2 == "v2"
    assert registry.get_active_strategy().strategy_id == "strat-strong"
