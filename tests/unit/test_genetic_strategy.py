"""
Unit tests for genetic strategy representation, operators, and registry.
"""

from pathlib import Path
import pytest
from evoagent.learning.genetic import (
    GeneticStrategyMutator,
    PromptStyle,
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


def test_tournament_generation_runs_against_frozen_memory(temp_repo: Path):
    from evoagent.domain.models import ExecutionBudget, Task
    from evoagent.evaluation.evaluator import IndependentEvaluator
    from evoagent.learning.evolution import EvolutionaryTournament
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

    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    store = InMemoryMemoryStore()
    patch_tool = PatchTool(root_path=temp_repo)
    sm = TaskStateMachine(
        planner=TaskPlanner(ExperienceRetriever(store)),
        repair_agent=RepairAgent(HeuristicCodingModel(), patch_tool),
        evaluator=IndependentEvaluator(test_runner),
        memory_store=store,
        patch_tool=patch_tool,
        git_inspector=GitInspector(sandbox),
        test_runner=test_runner,
    )
    task = Task(
        task_id="parser-empty-input",
        repository_path=temp_repo,
        issue_description="first_field crashes with IndexError on empty input",
        targeted_test="tests/test_parser.py -k test_empty_input",
        regression_test="tests/test_parser.py",
        budget=ExecutionBudget(max_attempts=2, timeout_seconds=30),
        metadata={"target_file": "parser.py"},
    )

    tournament = EvolutionaryTournament(sm, population_size=2)
    champion = tournament.run_generation([task], generation_idx=1)

    assert champion.fitness_score > 0.0
    assert len(tournament.population) == 2
    # Evaluating strategies must not write episodes that later chromosomes could retrieve
    assert store.get_all_episodes() == []
