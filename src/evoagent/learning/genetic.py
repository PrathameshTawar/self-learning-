"""
Genetic strategy representation, mutation, crossover, and fitness evaluation (Layer B).
"""

from enum import Enum
from pathlib import Path
import random
from typing import Optional
import uuid
from pydantic import BaseModel, Field


class PromptStyle(str, Enum):
    """Prompt template variants explored through strategy evolution."""

    MINIMAL_DIFF = "minimal_diff"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DEFENSIVE_BOUNDARY = "defensive_boundary"
    HYPOTHESIS_FIRST = "hypothesis_first"


class RetrievalPolicy(str, Enum):
    """Memory retrieval search strategies."""

    SIGNATURE_MATCH = "signature_match"
    BM25_SEMANTIC = "bm25_semantic"
    ERROR_TYPE_ONLY = "error_type_only"


class StrategyChromosome(BaseModel):
    """
    Genetic representation of an EvoAgent repair strategy policy.
    Evolved across generations on held-out benchmark tasks.
    """

    strategy_id: str = Field(default_factory=lambda: f"strat-{uuid.uuid4().hex[:8]}")
    prompt_style: PromptStyle = PromptStyle.MINIMAL_DIFF
    retrieval_top_k: int = Field(default=3, ge=1, le=8)
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.SIGNATURE_MATCH
    max_repair_attempts: int = Field(default=3, ge=1, le=6)
    max_patch_lines: int = Field(default=150, ge=30, le=500)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    enable_multi_file_context: bool = True
    generation: int = 0
    fitness_score: float = 0.0

    def compute_fitness(
        self,
        resolution_rate: float,
        regression_free_rate: float,
        avg_duration_seconds: float,
        avg_patch_lines: float = 5.0,
    ) -> float:
        """
        Multiobjective fitness function rewarding high repair accuracy and low regressions,
        penalizing excessive duration and bloated patches.
        """
        res_weight = 0.50
        reg_weight = 0.35
        dur_penalty = 0.10 * min(1.0, avg_duration_seconds / 60.0)
        size_penalty = 0.05 * min(1.0, avg_patch_lines / 100.0)

        score = (
            (resolution_rate / 100.0) * res_weight
            + (regression_free_rate / 100.0) * reg_weight
            - dur_penalty
            - size_penalty
        )
        self.fitness_score = round(max(0.0, score), 4)
        return self.fitness_score


class GeneticStrategyMutator:
    """Applies genetic operators (mutation and crossover) to strategy populations."""

    @staticmethod
    def mutate(parent: StrategyChromosome, mutation_rate: float = 0.3) -> StrategyChromosome:
        """Create a mutated offspring with perturbed hyperparameters across all genes."""
        mutated = parent.model_copy(deep=True)
        mutated.strategy_id = f"strat-{uuid.uuid4().hex[:8]}"
        mutated.generation = parent.generation + 1

        if random.random() < mutation_rate:
            mutated.prompt_style = random.choice(list(PromptStyle))

        if random.random() < mutation_rate:
            mutated.retrieval_top_k = max(1, min(8, parent.retrieval_top_k + random.choice([-1, 1])))

        if random.random() < mutation_rate:
            mutated.retrieval_policy = random.choice(list(RetrievalPolicy))

        if random.random() < mutation_rate:
            mutated.max_repair_attempts = max(1, min(6, parent.max_repair_attempts + random.choice([-1, 1])))

        if random.random() < mutation_rate:
            mutated.max_patch_lines = max(30, min(500, parent.max_patch_lines + random.choice([-25, 25])))

        if random.random() < mutation_rate:
            mutated.temperature = round(max(0.0, min(1.0, parent.temperature + random.choice([-0.1, 0.1]))), 2)

        if random.random() < mutation_rate:
            mutated.enable_multi_file_context = not parent.enable_multi_file_context

        return mutated

    @staticmethod
    def crossover(parent_a: StrategyChromosome, parent_b: StrategyChromosome) -> StrategyChromosome:
        """Combine dominant genes from two high-fitness strategy parents across all genes."""
        return StrategyChromosome(
            strategy_id=f"strat-{uuid.uuid4().hex[:8]}",
            prompt_style=parent_a.prompt_style if random.random() > 0.5 else parent_b.prompt_style,
            retrieval_top_k=random.choice([parent_a.retrieval_top_k, parent_b.retrieval_top_k]),
            retrieval_policy=parent_a.retrieval_policy if random.random() > 0.5 else parent_b.retrieval_policy,
            max_repair_attempts=max(parent_a.max_repair_attempts, parent_b.max_repair_attempts),
            max_patch_lines=random.choice([parent_a.max_patch_lines, parent_b.max_patch_lines]),
            temperature=round(random.choice([parent_a.temperature, parent_b.temperature]), 2),
            enable_multi_file_context=parent_a.enable_multi_file_context or parent_b.enable_multi_file_context,
            generation=max(parent_a.generation, parent_b.generation) + 1,
        )
