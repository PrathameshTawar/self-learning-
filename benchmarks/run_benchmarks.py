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
import statistics
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
from evoagent.evaluation.metrics import BenchmarkMetricsCalculator, mcnemar_exact_p
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import IMemoryStore, InMemoryMemoryStore, SQLiteMemoryStore
from evoagent.models.base import IModelBackend
from evoagent.models.factory import LLM_PROVIDERS, create_model_backend
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


def load_task(task_file: Path, max_attempts: Optional[int] = None, file_hints: bool = True) -> Task:
    """
    Build a Task from task.json, optionally overriding its attempt budget. With file_hints=False the
    metadata.target_file hint is dropped, so the planner must localize the defect from test output.
    """
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = dict(data.get("metadata", {}))
    if not file_hints:
        metadata.pop("target_file", None)

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
        metadata=metadata,
    )


def run_tasks(
    task_files: List[Path],
    model: IModelBackend,
    memory_store: IMemoryStore,
    max_attempts: Optional[int] = None,
    persist_episodes: bool = True,
    show_progress: bool = False,
    trace_dir: Optional[Path] = None,
    file_hints: bool = True,
) -> List[TaskResult]:
    """Run tasks in throwaway workspaces; persist_episodes=False evaluates against frozen memory."""
    sandbox = LocalSubprocessSandbox()
    test_runner = TestRunnerTool(sandbox)
    evaluator = IndependentEvaluator(test_runner)
    planner = TaskPlanner(ExperienceRetriever(memory_store))
    git_inspector = GitInspector(sandbox)

    results = []
    for task_file in task_files:
        task_obj = load_task(task_file, max_attempts, file_hints)
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
    if isinstance(model, HeuristicCodingModel):
        console.print(
            "[yellow]Model: offline heuristic mock. Its hard-coded rules only match the seeded train bugs, "
            "so these numbers say nothing about learning. Use --provider ollama for a real evaluation.[/]\n"
        )


def run_held_out_comparison(
    benchmark_root: Path,
    model: IModelBackend,
    results_path: Path,
    trace_dir: Optional[Path] = None,
    trials: int = 1,
    file_hints: bool = True,
) -> None:
    """Run the three-way ablation on held-out tasks `trials` times and report pooled metrics."""
    train_files = load_task_split(benchmark_root, "train")
    held_out_files = load_task_split(benchmark_root, "held_out")
    if not held_out_files or not train_files:
        console.print("[red]split.json must list both train and held-out tasks![/]")
        return

    trials = max(1, trials)
    console.print(Panel("[bold green]Evaluating Held-Out Benchmark Split across Ablation Conditions[/]"))
    console.print(
        f"Train tasks: {len(train_files)} | Held-out tasks: {len(held_out_files)} | Trials: {trials} | "
        f"File hints: {'on' if file_hints else 'off'}\n"
    )
    warn_if_mock(model)

    names = {
        "A": "Zero-Shot (No Memory, 1 Attempt)",
        "B": "Feedback Loop (No Memory, 3 Attempts)",
        "C": "Train-Split Memory + Feedback (3 Attempts)",
    }
    runs: Dict[str, List[List[TaskResult]]] = {cond_id: [] for cond_id in names}
    learned_episodes: List[int] = []

    for trial in range(1, trials + 1):
        console.print(f"[bold]Trial {trial}/{trials}[/]")
        # A: zero-shot baseline. Nothing is persisted, so no task can learn from an earlier held-out task.
        console.print("[cyan]  A: zero-shot (no memory, 1 attempt)...[/]")
        runs["A"].append(run_tasks(
            held_out_files, model, InMemoryMemoryStore(), max_attempts=1, persist_episodes=False,
            trace_dir=trace_dir, file_hints=file_hints,
        ))

        console.print("[cyan]  B: retry feedback (no memory, 3 attempts)...[/]")
        runs["B"].append(run_tasks(
            held_out_files, model, InMemoryMemoryStore(), max_attempts=3, persist_episodes=False,
            trace_dir=trace_dir, file_hints=file_hints,
        ))

        # C: memory built from the train split only, then frozen while evaluating held-out tasks.
        # Reusing an old memory DB would leak held-out solutions from earlier runs into the "learning" score.
        console.print("[cyan]  C: learning on train split, then held-out with frozen memory...[/]")
        learned_memory = InMemoryMemoryStore()
        run_tasks(train_files, model, learned_memory, persist_episodes=True, trace_dir=trace_dir, file_hints=file_hints)
        learned_episodes.append(len(learned_memory.get_accepted_episodes()))
        runs["C"].append(run_tasks(
            held_out_files, model, learned_memory, max_attempts=3, persist_episodes=False,
            trace_dir=trace_dir, file_hints=file_hints,
        ))

    # Pooled metrics over every trial, plus the spread of the per-trial resolution rates
    pooled = {
        cond_id: BenchmarkMetricsCalculator.calculate_metrics([res for trial in trial_runs for res in trial])
        for cond_id, trial_runs in runs.items()
    }
    per_trial = {
        cond_id: [BenchmarkMetricsCalculator.calculate_metrics(trial)["resolution_rate_percent"] for trial in trial_runs]
        for cond_id, trial_runs in runs.items()
    }

    def format_rate(cond_id: str) -> str:
        rate = f"{pooled[cond_id]['resolution_rate_percent']}%"
        return rate if trials == 1 else f"{rate} ± {statistics.stdev(per_trial[cond_id]):.1f}"

    table = Table(title="EvoAgent Held-Out Benchmark: Self-Learning Ablation Table")
    table.add_column("Evaluation Configuration", style="cyan")
    table.add_column("Tasks", justify="center")
    table.add_column("Resolution Rate" if trials == 1 else "Resolution (mean ± sd)", style="bold yellow", justify="center")
    table.add_column("Regression-Free", justify="center")
    table.add_column("Avg Attempts", justify="center")
    table.add_column("Delta vs A", style="bold green", justify="center")

    base_rate = pooled["A"]["resolution_rate_percent"]
    for cond_id, name in names.items():
        metrics = pooled[cond_id]
        delta = metrics["resolution_rate_percent"] - base_rate
        table.add_row(
            f"{cond_id}: {name}",
            str(len(held_out_files)) if trials == 1 else f"{len(held_out_files)} x {trials}",
            format_rate(cond_id),
            format_regression_free(metrics),
            str(metrics["average_attempts_per_task"]),
            "Baseline" if cond_id == "A" else f"{delta:+.1f} pts",
        )
    console.print("\n", table)

    # Paired B vs C comparison per (trial, task): only tasks solved by exactly one of them count
    paired = [(b.is_successful, c.is_successful) for tb, tc in zip(runs["B"], runs["C"]) for b, c in zip(tb, tc)]
    b_only = sum(1 for b, c in paired if b and not c)
    c_only = sum(1 for b, c in paired if c and not b)
    p_value = mcnemar_exact_p(b_only, c_only)
    memory_effect = pooled["C"]["resolution_rate_percent"] - pooled["B"]["resolution_rate_percent"]
    console.print(
        f"Memory effect (C - B): [bold]{memory_effect:+.1f} pts[/] | solved only by C: {c_only}, "
        f"only by B: {b_only} | exact McNemar p = {p_value:.3f}"
    )
    console.print(f"Verified train episodes available to C: {statistics.fmean(learned_episodes):.1f} per trial")
    if not any(learned_episodes):
        console.print("[yellow]No train task was solved, so condition C had nothing to retrieve.[/]")
    if p_value >= 0.05:
        console.print("[dim]Not statistically significant (p >= 0.05). Add trials or held-out tasks before claiming an effect.[/]")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": getattr(model, "model_name", type(model).__name__),
                "trials": trials,
                "file_hints": file_hints,
                "train_tasks": [load_task(f).task_id for f in train_files],
                "held_out_tasks": [load_task(f).task_id for f in held_out_files],
                "learned_episodes_per_trial": learned_episodes,
                "memory_effect_points": round(memory_effect, 2),
                "paired_b_vs_c": {"solved_only_by_b": b_only, "solved_only_by_c": c_only, "mcnemar_p": round(p_value, 4)},
                "conditions": [
                    {
                        "id": cond_id,
                        "name": name,
                        "total_tasks": len(held_out_files),
                        "resolution_rate_percent": pooled[cond_id]["resolution_rate_percent"],
                        "resolution_rate_per_trial": per_trial[cond_id],
                        "solved_tasks_per_trial": [
                            [res.task_id for res in trial if res.is_successful] for trial in runs[cond_id]
                        ],
                        "regression_free_rate_percent": (
                            pooled[cond_id]["regression_free_rate_percent"]
                            if pooled[cond_id]["targeted_pass_candidates"]
                            else None
                        ),
                        "average_attempts_per_task": pooled[cond_id]["average_attempts_per_task"],
                        "average_duration_seconds": pooled[cond_id]["average_duration_seconds"],
                    }
                    for cond_id, name in names.items()
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
    file_hints: bool = True,
) -> None:
    """Discover and execute benchmark tasks for given split."""
    task_files = load_task_split(benchmark_root, split)
    if not task_files:
        console.print(f"[red]No benchmark task.json files found for split: {split}![/]")
        return

    console.print(Panel(f"[bold green]Starting EvoAgent Benchmark Suite [{split}]: {len(task_files)} task(s) found[/]"))
    warn_if_mock(model)

    results = run_tasks(
        task_files, model, SQLiteMemoryStore(memory_db_path), show_progress=True, trace_dir=trace_dir,
        file_hints=file_hints,
    )
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
    parser.add_argument(
        "--benchmark-root",
        default=str(Path(__file__).parent),
        help="Directory with split.json and tasks/ (e.g. benchmarks/quixbugs after running import_quixbugs.py)",
    )
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
        "--request-timeout", type=float, default=settings.request_timeout_seconds, help="Seconds per model response"
    )
    parser.add_argument("--trials", type=int, default=1, help="Repeat the ablation N times (LLMs are nondeterministic)")
    parser.add_argument(
        "--no-file-hints",
        action="store_true",
        help="Drop metadata.target_file so the agent must find the defective file itself",
    )
    parser.add_argument(
        "--trace-dir",
        default=str(Path(__file__).parent.parent / settings.trace_dir),
        help="Directory for per-attempt JSON traces; pass an empty string to disable",
    )
    args = parser.parse_args()

    suite_dir = Path(args.benchmark_root)
    data_dir = Path(__file__).parent.parent / "data"
    traces = Path(args.trace_dir) if args.trace_dir else None
    model_backend = create_model_backend(
        args.provider, args.model, args.base_url, settings.api_key, timeout=args.request_timeout
    )

    if args.eval_comparison:
        run_held_out_comparison(
            suite_dir, model_backend, data_dir / "ablation_results.json", trace_dir=traces, trials=args.trials,
            file_hints=not args.no_file_hints,
        )
    else:
        run_benchmark_suite(
            suite_dir, data_dir / "benchmark_memory.db", model_backend, split=args.split, trace_dir=traces,
            file_hints=not args.no_file_hints,
        )
