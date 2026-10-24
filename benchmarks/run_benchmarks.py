"""
Automated benchmark runner for evaluating EvoAgent repair resolution and learning.
Supports train/held-out task splits and honest ablation comparisons:
- No Memory (1 Attempt Zero-Shot)
- Multi-Attempt with Feedback (No Memory, 3 Attempts)
- Memory-Guided Self-Learning (Episodic Memory + Multi-Attempt Feedback)
"""

import argparse
import json
from pathlib import Path
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoagent.domain.models import ExecutionBudget, Task
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore, SQLiteMemoryStore
from evoagent.models.heuristic_model import HeuristicCodingModel
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox.local_runner import LocalSubprocessSandbox
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.test_tools import TestRunnerTool

console = Console()


def load_task_split(benchmark_root: Path, split_name: str) -> list[Path]:
    """Load task.json paths filtered by train/held_out split."""
    split_file = benchmark_root / "split.json"
    tasks_dir = benchmark_root / "tasks"

    if split_name in ("train", "held_out") and split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        task_ids = split_data.get(f"{split_name}_tasks", [])
        matched = []
        for tid in task_ids:
            # Match folder by task_id or directory name
            for tpath in tasks_dir.glob("*/task.json"):
                with open(tpath, "r", encoding="utf-8") as tf:
                    tdata = json.load(tf)
                if tdata.get("task_id") == tid or tpath.parent.name == tid:
                    matched.append(tpath)
                    break
        return matched

    return sorted(tasks_dir.glob("*/task.json"))


def evaluate_tasks_under_condition(
    task_files: list[Path],
    enable_memory: bool,
    max_attempts: int,
    memory_db_path: Path,
) -> list:
    """Run tasks in a clean throwaway environment under specific ablation parameters."""
    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)

    if enable_memory and memory_db_path.exists():
        memory_store = SQLiteMemoryStore(memory_db_path)
    else:
        memory_store = InMemoryMemoryStore()

    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)
    model = HeuristicCodingModel()
    git_inspector = GitInspector(sandbox)

    results = []

    for task_file in task_files:
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
                max_attempts=max_attempts,
                timeout_seconds=data.get("timeout_seconds", 30),
            ),
            metadata=data.get("metadata", {}),
        )

        res = sm.execute_task(task_obj)
        results.append(res)

    return results


def run_held_out_comparison(benchmark_root: Path, memory_db_path: Path) -> None:
    """Run three-way ablation on held-out tasks and present comparative metrics table."""
    held_out_files = load_task_split(benchmark_root, "held_out")
    if not held_out_files:
        console.print("[red]No held-out benchmark tasks found in split.json![/]")
        return

    console.print(Panel("[bold green]Evaluating Held-Out Benchmark Split across Ablation Conditions[/]"))
    console.print(f"Found {len(held_out_files)} held-out evaluation tasks.\n")

    # Condition A: Zero-Shot Baseline (No Memory, 1 Attempt)
    console.print("[cyan][1/3] Running Zero-Shot Baseline (No Memory, 1 Attempt)...[/]")
    res_a = evaluate_tasks_under_condition(held_out_files, enable_memory=False, max_attempts=1, memory_db_path=memory_db_path)
    metrics_a = BenchmarkMetricsCalculator.calculate_metrics(res_a)

    # Condition B: Multi-Attempt with Feedback (No Memory, 3 Attempts)
    console.print("[cyan][2/3] Running Multi-Attempt with Retry Feedback (No Memory, 3 Attempts)...[/]")
    res_b = evaluate_tasks_under_condition(held_out_files, enable_memory=False, max_attempts=3, memory_db_path=memory_db_path)
    metrics_b = BenchmarkMetricsCalculator.calculate_metrics(res_b)

    # Condition C: Memory-Guided Self-Learning (Episodic Memory + Feedback, 3 Attempts)
    console.print("[cyan][3/3] Running Memory-Guided Self-Learning (Episodic Memory + Feedback)...[/]")
    res_c = evaluate_tasks_under_condition(held_out_files, enable_memory=True, max_attempts=3, memory_db_path=memory_db_path)
    metrics_c = BenchmarkMetricsCalculator.calculate_metrics(res_c)

    # Render comparative table
    table = Table(title="EvoAgent Held-Out Benchmark: Self-Learning Ablation Table")
    table.add_column("Evaluation Configuration", style="cyan")
    table.add_column("Tasks", justify="center")
    table.add_column("Resolution Rate", style="bold yellow", justify="center")
    table.add_column("Regression-Free", justify="center")
    table.add_column("Avg Attempts", justify="center")
    table.add_column("Self-Learning Delta", style="bold green", justify="center")

    base_rate = metrics_a["resolution_rate_percent"]
    table.add_row(
        "A: Zero-Shot (No Memory, 1 Attempt)",
        str(metrics_a["total_tasks"]),
        f"{metrics_a['resolution_rate_percent']}%",
        f"{metrics_a['regression_free_rate_percent']}%",
        "1.0",
        "Baseline (0.0%)",
    )
    delta_b = metrics_b["resolution_rate_percent"] - base_rate
    table.add_row(
        "B: Feedback Loop (No Memory, 3 Attempts)",
        str(metrics_b["total_tasks"]),
        f"{metrics_b['resolution_rate_percent']}%",
        f"{metrics_b['regression_free_rate_percent']}%",
        str(metrics_b["average_attempts_per_task"]),
        f"{delta_b:+.1f}% (Retry Feedback)",
    )
    delta_c = metrics_c["resolution_rate_percent"] - base_rate
    table.add_row(
        "C: Full Learning (Episodic Memory + 3 Attempts)",
        str(metrics_c["total_tasks"]),
        f"{metrics_c['resolution_rate_percent']}%",
        f"{metrics_c['regression_free_rate_percent']}%",
        str(metrics_c["average_attempts_per_task"]),
        f"{delta_c:+.1f}% (Memory + Feedback)",
    )

    console.print("\n", table)


def run_benchmark_suite(benchmark_root: Path, memory_db_path: Path, split: str = "all") -> None:
    """Discover and execute benchmark tasks for given split."""
    task_files = load_task_split(benchmark_root, split)
    if not task_files:
        console.print(f"[red]No benchmark task.json files found for split: {split}![/]")
        return

    console.print(Panel(f"[bold green]Starting EvoAgent Benchmark Suite [{split}]: {len(task_files)} task(s) found[/]"))

    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)
    memory_store = SQLiteMemoryStore(memory_db_path)
    retriever = ExperienceRetriever(memory_store)
    planner = TaskPlanner(retriever)
    model = HeuristicCodingModel()
    git_inspector = GitInspector(sandbox)

    results = []

    for task_file in task_files:
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

    metrics = BenchmarkMetricsCalculator.calculate_metrics(results)

    table = Table(title=f"EvoAgent Benchmark Performance Summary [{split.upper()}]")
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
    parser = argparse.ArgumentParser(description="EvoAgent Benchmark Runner")
    parser.add_argument("--split", choices=["all", "train", "held_out"], default="all")
    parser.add_argument("--eval-comparison", action="store_true", help="Run 3-way ablation comparison on held-out tasks")
    args = parser.parse_args()

    suite_dir = Path(__file__).parent
    db_file = Path(__file__).parent.parent / "data" / "benchmark_memory.db"

    if args.eval_comparison:
        run_held_out_comparison(suite_dir, db_file)
    else:
        run_benchmark_suite(suite_dir, db_file, split=args.split)
