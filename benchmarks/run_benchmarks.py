"""
Automated benchmark runner for evaluating EvoAgent repair resolution and learning.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoagent.domain.models import ExecutionBudget, Task
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import SQLiteMemoryStore
from evoagent.models.heuristic_model import HeuristicCodingModel
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool

console = Console()


def run_benchmark_suite(benchmark_root: Path, memory_db_path: Path) -> None:
    """Discover and execute all benchmark tasks, displaying aggregated metrics."""
    tasks_dir = benchmark_root / "tasks"
    task_files = list(tasks_dir.glob("*/task.json"))

    if not task_files:
        console.print("[red]No benchmark task.json files found![/]")
        return

    console.print(Panel(f"[bold green]Starting EvoAgent Benchmark Suite: {len(task_files)} task(s) found[/]"))

    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)
    memory_store = SQLiteMemoryStore(memory_db_path)
    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)
    model = HeuristicCodingModel()
    git_inspector = GitInspector(sandbox)

    results = []

    for task_file in sorted(task_files):
        with open(task_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        repo_dir = task_file.parent / data.get("repo_dir", "code")
        patch_tool = PatchTool(root_path=repo_dir)
        repair_agent = RepairAgent(model, patch_tool)

        sm = TaskStateMachine(
            planner=planner,
            repair_agent=repair_agent,
            evaluator=evaluator,
            memory_store=memory_store,
            patch_tool=patch_tool,
            git_inspector=git_inspector,
            test_runner=test_runner,
        )

        task_obj = Task(
            task_id=data.get("task_id", task_file.parent.name),
            repository_path=repo_dir,
            issue_description=data.get("issue_description", ""),
            targeted_test=data.get("targeted_test", ""),
            regression_test=data.get("regression_test"),
            budget=ExecutionBudget(
                max_attempts=data.get("max_attempts", 3),
                timeout_seconds=data.get("timeout_seconds", 30),
            ),
            metadata=data.get("metadata", {}),
        )

        console.print(f"[>] Executing task: [cyan]{task_obj.task_id}[/] ... ", end="")
        res = sm.execute_task(task_obj)
        results.append(res)

        if res.is_successful:
            console.print("[bold green]PASSED[/]")
        else:
            console.print("[bold red]FAILED[/]")

    # Compute and display aggregate metrics
    metrics = BenchmarkMetricsCalculator.calculate_metrics(results)

    table = Table(title="EvoAgent Benchmark Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Measured Value", style="bold yellow")
    table.add_column("Engineering Target", style="green")

    table.add_row("Total Tasks", str(metrics["total_tasks"]), "N/A")
    table.add_row("Task Resolution Rate", f"{metrics['resolution_rate_percent']}%", ">= 50%")
    table.add_row("Regression-Free Rate", f"{metrics['regression_free_rate_percent']}%", ">= 95%")
    table.add_row("Patch Validity Rate", f"{metrics['patch_validity_rate_percent']}%", ">= 95%")
    table.add_row("Avg Task Duration", f"{metrics['average_duration_seconds']}s", "< 30s")

    console.print("\n", table)


if __name__ == "__main__":
    suite_dir = Path(__file__).parent
    db_file = Path(__file__).parent.parent / "data" / "benchmark_memory.db"
    run_benchmark_suite(suite_dir, db_file)
