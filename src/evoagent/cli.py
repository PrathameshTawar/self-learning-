"""
Command Line Interface for EvoAgent using Click and Rich.
"""

import json
from pathlib import Path
from typing import List
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evoagent.config import settings
from evoagent.domain.models import ExecutionBudget, Task
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.learning.dataset import DatasetCurator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import SQLiteMemoryStore
from evoagent.models.factory import create_model_backend
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool

console = Console()


def _build_state_machine(memory_path: Path) -> TaskStateMachine:
    """Wire dependencies for the state machine."""
    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)
    memory_store = SQLiteMemoryStore(memory_path)
    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)

    model_backend = create_model_backend(
        settings.model_provider, settings.model_name, settings.api_base_url, settings.api_key,
        timeout=settings.request_timeout_seconds,
    )

    git_inspector = GitInspector(sandbox)
    patch_tool = PatchTool(root_path=Path("."))
    repair_agent = RepairAgent(model_backend, patch_tool)

    return TaskStateMachine(
        planner=planner,
        repair_agent=repair_agent,
        evaluator=evaluator,
        memory_store=memory_store,
        patch_tool=patch_tool,
        git_inspector=git_inspector,
        test_runner=test_runner,
        trace_dir=settings.trace_dir,
    )


def _task_from_file(task_path: Path) -> Task:
    """Build a Task from a benchmark task.json file."""
    with open(task_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Task(
        task_id=data.get("task_id", task_path.stem),
        repository_path=task_path.parent / data.get("repo_dir", "code"),
        issue_description=data.get("issue_description", ""),
        targeted_test=data.get("targeted_test", ""),
        regression_test=data.get("regression_test"),
        budget=ExecutionBudget(
            max_attempts=data.get("max_attempts", 3),
            timeout_seconds=data.get("timeout_seconds", 60),
        ),
        metadata=data.get("metadata", {}),
    )


def _load_split_tasks(benchmark_root: Path, split_name: str) -> List[Task]:
    """Load the tasks listed under "<split_name>_tasks" in split.json."""
    split_file = benchmark_root / "split.json"
    if not split_file.exists():
        return []
    task_ids = set(json.loads(split_file.read_text(encoding="utf-8")).get(f"{split_name}_tasks", []))
    tasks = [_task_from_file(f) for f in sorted((benchmark_root / "tasks").glob("*/task.json"))]
    return [t for t in tasks if t.task_id in task_ids]


@click.group()
def main() -> None:
    """EvoAgent: A Self-Learning Autonomous Coding and Repair Agent."""
    pass


@main.command()
@click.option("--task", "-t", required=True, type=click.Path(exists=True), help="Path to task.json")
@click.option("--memory-db", default="data/evoagent_memory.db", help="Path to SQLite memory file")
def run(task: str, memory_db: str) -> None:
    """Run EvoAgent autonomous repair loop on a task file."""
    task_obj = _task_from_file(Path(task))

    console.print(Panel(f"[bold cyan]EvoAgent Running Task:[/] {task_obj.task_id}\n[yellow]Issue:[/] {task_obj.issue_description}"))

    sm = _build_state_machine(Path(memory_db))
    sm.patch_tool.root_path = task_obj.repository_path

    result = sm.execute_task(task_obj)

    if result.is_successful:
        console.print(f"[bold green][PASS] Task {result.task_id} COMPLETED SUCCESSFULLY![/]")
        if result.accepted_patch:
            console.print("\n[bold]Accepted Patch Diff:[/]")
            console.print(result.accepted_patch.diff_unified)
    else:
        console.print(f"[bold red][FAIL] Task {result.task_id} FAILED after {len(result.attempts)} attempts.[/]")

    console.print(f"Total Duration: {result.total_duration_seconds}s | Final State: {result.final_state.value}")


@main.command()
@click.option("--memory-db", default="data/evoagent_memory.db", help="Path to SQLite memory file")
def memory_list(memory_db: str) -> None:
    """List episodic memory records."""
    store = SQLiteMemoryStore(Path(memory_db))
    episodes = store.get_all_episodes(limit=25)

    if not episodes:
        console.print("[dim]No memory episodes recorded yet.[/]")
        return

    table = Table(title="EvoAgent Episodic Memory Store")
    table.add_column("Task ID", style="cyan")
    table.add_column("Failure Signature", style="red")
    table.add_column("Outcome", style="green")
    table.add_column("Lessons Learned", style="white")

    for ep in episodes:
        outcome_color = "green" if ep.outcome == "accepted" else "red"
        table.add_row(
            ep.task_id,
            ep.failure_signature[:40],
            f"[{outcome_color}]{ep.outcome}[/]",
            ep.lessons_learned[:60],
        )

    console.print(table)


@main.command()
@click.option("--output", "-o", default="data/curated_sft.jsonl", help="Output JSONL file path")
@click.option("--memory-db", default="data/evoagent_memory.db", help="Path to SQLite memory file")
@click.option("--trace-dir", default=str(settings.trace_dir), help="Run traces (preferred source of full prompts)")
def export_dataset(output: str, memory_db: str, trace_dir: str) -> None:
    """Export verified repairs as an SFT dataset, plus DPO pairs built from traces."""
    curator = DatasetCurator(SQLiteMemoryStore(Path(memory_db)))
    out_path = Path(output)
    rows = curator.export_dataset_jsonl(out_path, trace_dir=Path(trace_dir))
    dpo_path = out_path.with_name(f"{out_path.stem}_dpo.jsonl")
    pairs = curator.export_dpo_jsonl(dpo_path, Path(trace_dir))
    console.print(f"[bold green]Exported {rows} SFT rows to {out_path} and {pairs} DPO pairs to {dpo_path}[/]")


@main.command()
@click.option("--port", "-p", default=8080, help="Port to serve the dashboard on")
@click.option("--no-browser", is_flag=True, help="Do not open browser automatically")
def dashboard(port: int, no_browser: bool) -> None:
    """Launch the interactive EvoAgent web dashboard."""
    from evoagent.dashboard.server import start_dashboard
    console.print(Panel(f"[bold green]Launching EvoAgent Web Dashboard at http://localhost:{port}[/]"))
    start_dashboard(port=port, open_browser=not no_browser)


@main.command()
@click.option("--repo", "-r", default=".", help="Path to repository")
def symbols(repo: str) -> None:
    """Index and display AST symbols and cross-file references."""
    from evoagent.intelligence.symbol_graph import SymbolGraph
    graph = SymbolGraph(Path(repo))

    table = Table(title=f"EvoAgent AST Symbol Index ({len(graph.symbols_by_name)} symbols)")
    table.add_column("Symbol Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("File", style="green")
    table.add_column("Signature", style="white")

    for name, sym_list in list(graph.symbols_by_name.items())[:30]:
        for s in sym_list:
            table.add_row(s.name, s.symbol_type.value, s.file_path, s.signature[:50])

    console.print(table)


@main.command()
@click.option("--generations", "-g", default=2, help="Number of evolutionary tournament generations")
@click.option("--population", "-pop", default=4, help="Strategy population size")
@click.option("--benchmarks", default="benchmarks", help="Benchmark root containing split.json and tasks/")
def strategy_evolve(generations: int, population: int, benchmarks: str) -> None:
    """Evolve strategies on the train split; promote the champion only on held-out evidence."""
    from evoagent.learning.evolution import EvolutionaryTournament
    from evoagent.learning.registry import StrategyRegistry

    benchmark_root = Path(benchmarks)
    train_tasks = _load_split_tasks(benchmark_root, "train")
    held_out_tasks = _load_split_tasks(benchmark_root, "held_out")
    if not train_tasks or not held_out_tasks:
        console.print(f"[red]Need train and held-out tasks in {benchmark_root / 'split.json'}.[/]")
        return

    console.print(Panel(
        f"[bold magenta]Starting Strategy Evolutionary Tournament[/]\n"
        f"Generations: {generations} | Population: {population} | "
        f"Train tasks: {len(train_tasks)} | Held-out tasks: {len(held_out_tasks)}"
    ))

    sm = _build_state_machine(Path("data/evolution_memory.db"))
    tournament = EvolutionaryTournament(sm, population_size=population)
    champion = tournament.evolve(train_tasks, total_generations=generations)
    train_fitness = champion.fitness_score

    # Fitness measured on the tasks used for selection is optimistic; re-score on unseen tasks
    held_out_metrics = tournament.evaluate_strategy(champion, held_out_tasks)
    console.print(f"Champion {champion.strategy_id}: train fitness {train_fitness} | held-out fitness {champion.fitness_score}")
    console.print(f"Prompt Style: {champion.prompt_style.value} | Top-K: {champion.retrieval_top_k} | Multi-File: {champion.enable_multi_file_context}")

    registry = StrategyRegistry(Path("data/strategies.json"))
    try:
        v_id = registry.promote_strategy(
            champion,
            benchmark_evidence={
                "split": "held_out",
                "fitness": champion.fitness_score,
                "train_fitness": train_fitness,
                "resolution_rate_percent": held_out_metrics["resolution_rate_percent"],
                "tasks": [t.task_id for t in held_out_tasks],
            },
        )
    except ValueError as exc:
        console.print(f"[yellow]Champion not promoted:[/] {exc}")
        return

    console.print(f"\n[bold green]Champion promoted as version [bold cyan]{v_id}[/][/]")


if __name__ == "__main__":
    main()
