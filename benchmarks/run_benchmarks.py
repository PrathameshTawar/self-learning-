"""
Automated benchmark runner for evaluating EvoAgent repair resolution and learning.
Supports train/held-out task splits and honest ablation comparisons:
- No Memory (1 Attempt Zero-Shot)
- Multi-Attempt with Feedback (No Memory, 3 Attempts)
- Memory-Guided Self-Learning (memory learned on the train split only, frozen during held-out evaluation)
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoagent.config import settings
from evoagent.domain.models import ExecutionBudget, Task, TaskResult
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import IMemoryStore, InMemoryMemoryStore, SQLiteMemoryStore
from evoagent.models.base import IModelBackend
from evoagent.models.factory import LLM_PROVIDERS, create_model_backend
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


def load_task(task_file: Path, max_attempts: Optional[int] = None) -> Task:
    """Build a Task from task.json, optionally overriding its attempt budget."""
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Task(
        task_id=data.get("task_id", task_file.parent.name),
        repository_path=task_file.parent / data.get("repo_dir", "code"),
        issue_description=data.get("issue_description", ""),
        targeted_test=data.get("targeted_test", ""),
        regression_test=data.get("regression_test"),
        budget=ExecutionBudget(
            max_attempts=max_attempts or data.get("max_attempts", 3),
            timeout_seconds=data.get("timeout_seconds", 30),
        ),
        metadata=data.get("metadata", {}),
    )


def run_tasks(
    task_files: List[Path],
    model: IModelBackend,
    memory_store: IMemoryStore,
    max_attempts: Optional[int] = None,
    persist_episodes: bool = True,
    show_progress: bool = False,
    trace_dir: Optional[Path] = None,
) -> List[TaskResult]:
    """Run tasks in throwaway workspaces; persist_episodes=False evaluates against frozen memory."""
    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)
    planner = TaskPlanner(ExperienceRetriever(memory_store))
    git_inspector = GitInspector(sandbox)

    results = []
    for task_file in task_files:
        task_obj = load_task(task_file, max_attempts)
        patch_tool = PatchTool(root_path=task_obj.repository_path)
        sm = TaskStateMachine(
            planner=planner,
            repair_agent=RepairAgent(model, patch_tool),
            evaluator=evaluator,
            memory_store=memory_store,
            patch_tool=patch_tool,
            git_inspector=git_inspector,
            test_runner=test_runner,
            trace_dir=trace_dir,
        )

        if show_progress:
            console.print(f"[>] Executing task: [cyan]{task_obj.task_id}[/] ... ", end="")
        res = sm.execute_task(task_obj, persist_episode=persist_episodes)
        results.append(res)
        if show_progress:
            console.print("[bold green]PASSED[/]" if res.is_successful else f"[bold red]{res.final_state.value}[/]")

    return results


def format_regression_free(metrics: Dict[str, Any]) -> str:
    """Regression-free rate is undefined when no candidate patch passed its targeted test."""
    if metrics["targeted_pass_candidates"] == 0:
        return "n/a"
    return f"{metrics['regression_free_rate_percent']}%"


def warn_if_mock(model: IModelBackend) -> None:
    if not hasattr(model, "base_url"):
        console.print(
            "[yellow]Model: offline heuristic mock. Its hard-coded rules only match the seeded train bugs, "
            "so these numbers say nothing about learning. Use --provider ollama for a real evaluation.[/]\n"
        )


def run_held_out_comparison(
    benchmark_root: Path, model: IModelBackend, results_path: Path, trace_dir: Optional[Path] = None
) -> None:
    """Run three-way ablation on held-out tasks and present comparative metrics table."""
    train_files = load_task_split(benchmark_root, "train")
    held_out_files = load_task_split(benchmark_root, "held_out")
    if not held_out_files or not train_files:
        console.print("[red]split.json must list both train and held-out tasks![/]")
        return

    console.print(Panel("[bold green]Evaluating Held-Out Benchmark Split across Ablation Conditions[/]"))
    console.print(f"Train tasks: {len(train_files)} | Held-out tasks: {len(held_out_files)}\n")
    warn_if_mock(model)

    # Condition A: Zero-Shot Baseline (No Memory, 1 Attempt). Nothing is persisted, so no task
    # can learn from an earlier held-out task.
    console.print("[cyan][1/3] Running Zero-Shot Baseline (No Memory, 1 Attempt)...[/]")
    res_a = run_tasks(
        held_out_files, model, InMemoryMemoryStore(), max_attempts=1, persist_episodes=False, trace_dir=trace_dir
    )
    metrics_a = BenchmarkMetricsCalculator.calculate_metrics(res_a)

    # Condition B: Multi-Attempt with Feedback (No Memory, 3 Attempts)
    console.print("[cyan][2/3] Running Multi-Attempt with Retry Feedback (No Memory, 3 Attempts)...[/]")
    res_b = run_tasks(
        held_out_files, model, InMemoryMemoryStore(), max_attempts=3, persist_episodes=False, trace_dir=trace_dir
    )
    metrics_b = BenchmarkMetricsCalculator.calculate_metrics(res_b)

    # Condition C: memory built from the train split only, then frozen while evaluating held-out tasks.
    # Reusing an old memory DB would leak held-out solutions from earlier runs into the "learning" score.
    console.print("[cyan][3/3] Learning on train split, then evaluating held-out tasks with frozen memory...[/]")
    learned_memory = InMemoryMemoryStore()
    run_tasks(train_files, model, learned_memory, persist_episodes=True, trace_dir=trace_dir)
    learned_episodes = len(learned_memory.get_accepted_episodes())
    res_c = run_tasks(
        held_out_files, model, learned_memory, max_attempts=3, persist_episodes=False, trace_dir=trace_dir
    )
    metrics_c = BenchmarkMetricsCalculator.calculate_metrics(res_c)

    conditions = [
        ("A", "Zero-Shot (No Memory, 1 Attempt)", metrics_a),
        ("B", "Feedback Loop (No Memory, 3 Attempts)", metrics_b),
        ("C", "Train-Split Memory + Feedback (3 Attempts)", metrics_c),
    ]

    table = Table(title="EvoAgent Held-Out Benchmark: Self-Learning Ablation Table")
    table.add_column("Evaluation Configuration", style="cyan")
    table.add_column("Tasks", justify="center")
    table.add_column("Resolution Rate", style="bold yellow", justify="center")
    table.add_column("Regression-Free", justify="center")
    table.add_column("Avg Attempts", justify="center")
    table.add_column("Delta vs A", style="bold green", justify="center")

    base_rate = metrics_a["resolution_rate_percent"]
    for cond_id, name, metrics in conditions:
        delta = metrics["resolution_rate_percent"] - base_rate
        table.add_row(
            f"{cond_id}: {name}",
            str(metrics["total_tasks"]),
            f"{metrics['resolution_rate_percent']}%",
            format_regression_free(metrics),
            str(metrics["average_attempts_per_task"]),
            "Baseline" if cond_id == "A" else f"{delta:+.1f} pts",
        )

    console.print("\n", table)
    memory_effect = metrics_c["resolution_rate_percent"] - metrics_b["resolution_rate_percent"]
    console.print(
        f"Memory effect (C - B): [bold]{memory_effect:+.1f} pts[/] | "
        f"Verified train episodes available to C: {learned_episodes}"
    )
    if learned_episodes == 0:
        console.print("[yellow]No train task was solved, so condition C had nothing to retrieve.[/]")
    console.print(
        f"[dim]With {len(held_out_files)} held-out tasks each task moves the rate by "
        f"{100 / len(held_out_files):.0f} pts; differences this size are not statistically meaningful.[/]"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": getattr(model, "model_name", type(model).__name__),
                "train_tasks": [load_task(f).task_id for f in train_files],
                "held_out_tasks": [load_task(f).task_id for f in held_out_files],
                "learned_episodes": learned_episodes,
                "memory_effect_points": round(memory_effect, 2),
                "conditions": [
                    {
                        "id": cond_id,
                        "name": name,
                        "total_tasks": metrics["total_tasks"],
                        "resolution_rate_percent": metrics["resolution_rate_percent"],
                        "regression_free_rate_percent": (
                            metrics["regression_free_rate_percent"] if metrics["targeted_pass_candidates"] else None
                        ),
                        "average_attempts_per_task": metrics["average_attempts_per_task"],
                        "average_duration_seconds": metrics["average_duration_seconds"],
                    }
                    for cond_id, name, metrics in conditions
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"Results written to {results_path.as_posix()}")


def run_benchmark_suite(
    benchmark_root: Path,
    memory_db_path: Path,
    model: IModelBackend,
    split: str = "all",
    trace_dir: Optional[Path] = None,
) -> None:
    """Discover and execute benchmark tasks for given split."""
    task_files = load_task_split(benchmark_root, split)
    if not task_files:
        console.print(f"[red]No benchmark task.json files found for split: {split}![/]")
        return

    console.print(Panel(f"[bold green]Starting EvoAgent Benchmark Suite [{split}]: {len(task_files)} task(s) found[/]"))
    warn_if_mock(model)

    results = run_tasks(task_files, model, SQLiteMemoryStore(memory_db_path), show_progress=True, trace_dir=trace_dir)
    metrics = BenchmarkMetricsCalculator.calculate_metrics(results)

    table = Table(title=f"EvoAgent Benchmark Performance Summary [{split.upper()}]")
    table.add_column("Metric", style="cyan")
    table.add_column("Measured Value", style="bold yellow")
    table.add_column("Engineering Target", style="green")

    table.add_row("Total Tasks", str(metrics["total_tasks"]), "N/A")
    table.add_row("Task Resolution Rate", f"{metrics['resolution_rate_percent']}%", ">= 50%")
    table.add_row("Regression-Free Rate", format_regression_free(metrics), ">= 95%")
    table.add_row("Patch Validity Rate", f"{metrics['patch_validity_rate_percent']}%", ">= 95%")
    table.add_row("Avg Task Duration", f"{metrics['average_duration_seconds']}s", "< 30s")

    console.print("\n", table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoAgent Benchmark Runner")
    parser.add_argument("--split", choices=["all", "train", "held_out"], default="all")
    parser.add_argument("--eval-comparison", action="store_true", help="Run 3-way ablation comparison on held-out tasks")
    parser.add_argument(
        "--provider",
        choices=["heuristic", *LLM_PROVIDERS],
        default=settings.model_provider if settings.model_provider in LLM_PROVIDERS else "heuristic",
        help="Model backend (defaults to EVOAGENT_MODEL_PROVIDER)",
    )
    parser.add_argument("--model", default=settings.model_name, help="Model name for LLM providers")
    parser.add_argument("--base-url", default=settings.api_base_url, help="OpenAI-compatible endpoint")
    parser.add_argument(
        "--trace-dir",
        default=str(Path(__file__).parent.parent / settings.trace_dir),
        help="Directory for per-attempt JSON traces; pass an empty string to disable",
    )
    args = parser.parse_args()

    suite_dir = Path(__file__).parent
    data_dir = Path(__file__).parent.parent / "data"
    traces = Path(args.trace_dir) if args.trace_dir else None
    model_backend = create_model_backend(args.provider, args.model, args.base_url, settings.api_key)

    if args.eval_comparison:
        run_held_out_comparison(suite_dir, model_backend, data_dir / "ablation_results.json", trace_dir=traces)
    else:
        run_benchmark_suite(suite_dir, data_dir / "benchmark_memory.db", model_backend, split=args.split, trace_dir=traces)
