"""
Evolutionary tournament engine executing generational strategy evaluation.
"""

from typing import Any, Dict, List, Tuple
from evoagent.domain.models import Task, TaskResult
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator
from evoagent.learning.genetic import (
    GeneticStrategyMutator,
    PromptStyle,
    RetrievalPolicy,
    StrategyChromosome,
)
from evoagent.orchestration.state_machine import TaskStateMachine


class EvolutionaryTournament:
    """
    Executes generational tournaments to evolve optimal repair strategies.
    Evaluates chromosomes on benchmark suites, ranks fitness, and generates offspring.
    """

    def __init__(self, state_machine: TaskStateMachine, population_size: int = 4) -> None:
        self.state_machine = state_machine
        self.population_size = max(2, population_size)
        self.population: List[StrategyChromosome] = self._init_population()
        self.generation_history: List[Dict[str, Any]] = []

    def _init_population(self) -> List[StrategyChromosome]:
        """Seed diverse initial population."""
        return [
            StrategyChromosome(
                strategy_id="strat-default",
                prompt_style=PromptStyle.MINIMAL_DIFF,
                retrieval_top_k=3,
                retrieval_policy=RetrievalPolicy.SIGNATURE_MATCH,
                max_patch_lines=150,
                temperature=0.2,
                enable_multi_file_context=True,
            ),
            StrategyChromosome(
                strategy_id="strat-defensive",
                prompt_style=PromptStyle.DEFENSIVE_BOUNDARY,
                retrieval_top_k=5,
                retrieval_policy=RetrievalPolicy.BM25_SEMANTIC,
                max_patch_lines=120,
                temperature=0.1,
                enable_multi_file_context=True,
            ),
            StrategyChromosome(
                strategy_id="strat-cot",
                prompt_style=PromptStyle.CHAIN_OF_THOUGHT,
                retrieval_top_k=2,
                retrieval_policy=RetrievalPolicy.ERROR_TYPE_ONLY,
                max_patch_lines=180,
                temperature=0.3,
                enable_multi_file_context=False,
            ),
            StrategyChromosome(
                strategy_id="strat-hypothesis",
                prompt_style=PromptStyle.HYPOTHESIS_FIRST,
                retrieval_top_k=4,
                retrieval_policy=RetrievalPolicy.SIGNATURE_MATCH,
                max_patch_lines=200,
                temperature=0.2,
                enable_multi_file_context=True,
            ),
        ][: self.population_size]

    def run_generation(self, benchmark_tasks: List[Task], generation_idx: int) -> StrategyChromosome:
        """Evaluate current population on benchmark tasks with full gene wiring and compute fitness scores."""
        evaluated_chromosomes: List[Tuple[StrategyChromosome, Dict[str, Any]]] = []

        for chrom in self.population:
            results: List[TaskResult] = []
            for t in benchmark_tasks:
                # Fully pass chromosome to execute_task so all genes are active
                res = self.state_machine.execute_task(t, strategy=chrom)
                results.append(res)

            metrics = BenchmarkMetricsCalculator.calculate_metrics(results)
            chrom.compute_fitness(
                resolution_rate=metrics["resolution_rate_percent"],
                regression_free_rate=metrics["regression_free_rate_percent"],
                avg_duration_seconds=metrics["average_duration_seconds"],
            )

            evaluated_chromosomes.append((chrom, metrics))

        # Sort descending by fitness
        evaluated_chromosomes.sort(key=lambda x: x[0].fitness_score, reverse=True)
        champion = evaluated_chromosomes[0][0]

        # Record generation summary
        self.generation_history.append({
            "generation": generation_idx,
            "champion_id": champion.strategy_id,
            "best_fitness": champion.fitness_score,
            "prompt_style": champion.prompt_style.value,
            "retrieval_policy": champion.retrieval_policy.value,
            "temperature": champion.temperature,
            "enable_multi_file_context": champion.enable_multi_file_context,
            "population_fitnesses": [c[0].fitness_score for c in evaluated_chromosomes],
        })

        # Produce next generation (Elitism + Mutation + Crossover)
        survivors = [c[0] for c in evaluated_chromosomes[: len(evaluated_chromosomes) // 2]]
        next_gen: List[StrategyChromosome] = [champion.model_copy(deep=True)]

        while len(next_gen) < self.population_size:
            p1 = survivors[0]
            p2 = survivors[min(1, len(survivors) - 1)]
            child = GeneticStrategyMutator.crossover(p1, p2)
            mutated_child = GeneticStrategyMutator.mutate(child, mutation_rate=0.4)
            mutated_child.generation = generation_idx + 1
            next_gen.append(mutated_child)

        self.population = next_gen
        return champion

    def evolve(self, benchmark_tasks: List[Task], total_generations: int = 2) -> StrategyChromosome:
        """Run full evolutionary loop across multiple generations."""
        best_overall = self.population[0]
        for g in range(1, total_generations + 1):
            gen_champ = self.run_generation(benchmark_tasks, generation_idx=g)
            if gen_champ.fitness_score > best_overall.fitness_score:
                best_overall = gen_champ

        return best_overall
