"""
Command Line Interface for EvoAgent using Click and Rich.
"""

import json
from pathlib import Path
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
from evoagent.models.heuristic_model import HeuristicCodingModel
from evoagent.models.open_model import OpenAICompatibleModel
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

    # Choose model backend based on settings
    if settings.model_provider == "openai_compatible" or settings.model_provider == "ollama":
        model_backend = OpenAICompatibleModel(
            base_url=settings.api_base_url,
            model_name=settings.model_name,
            api_key=settings.api_key,
        )
    else:
        model_backend = HeuristicCodingModel()

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
    )


@click.group()
def main() -> None:
    """EvoAgent: A Self-Learning Autonomous Coding and Repair Agent."""
    pass


@main.command()
@click.option("--task", "-t", required=True, type=click.Path(exists=True), help="Path to task.json")
@click.option("--memory-db", default="data/evoagent_memory.db", help="Path to SQLite memory file")
def run(task: str, memory_db: str) -> None:
    """Run EvoAgent autonomous repair loop on a task file."""
    task_path = Path(task)
    with open(task_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_obj = Task(
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

    console.print(Panel(f"[bold cyan]EvoAgent Running Task:[/] {task_obj.task_id}\n[yellow]Issue:[/] {task_obj.issue_description}"))

    sm = _build_state_machine(Path(memory_db))
    sm.patch_tool.root_path = task_obj.repository_path

    result = sm.execute_task(task_obj)

    if result.is_successful:
        console.print(f"[bold green]✔ Task {result.task_id} COMPLETED SUCCESSFULLY![/]")
        if result.accepted_patch:
            console.print("\n[bold]Accepted Patch Diff:[/]")
            console.print(result.accepted_patch.diff_unified)
    else:
        console.print(f"[bold red]✘ Task {result.task_id} FAILED after {len(result.attempts)} attempts.[/]")

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
def export_dataset(output: str, memory_db: str) -> None:
    """Export verified repairs into SFT training dataset."""
    store = SQLiteMemoryStore(Path(memory_db))
    curator = DatasetCurator(store)
    out_path = Path(output)
    curator.export_dataset_jsonl(out_path)
    console.print(f"[bold green]Successfully exported training dataset to {out_path}[/]")


if __name__ == "__main__":
    main()
