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
from evoagent.domain.models import Task
from evoagent.learning.dataset import DatasetCurator
from evoagent.memory.store import SQLiteMemoryStore
from evoagent.models.factory import create_model_backend
from evoagent.orchestration.factory import build_state_machine, task_from_file
from evoagent.orchestration.state_machine import TaskStateMachine
from evoagent.sandbox import create_sandbox

console = Console()


def _build_state_machine(memory_path: Path) -> TaskStateMachine:
    """Wire dependencies for the state machine."""
    model_backend = create_model_backend(
        settings.model_provider, settings.model_name, settings.api_base_url, settings.api_key,
        timeout=settings.request_timeout_seconds,
    )
    return build_state_machine(
        SQLiteMemoryStore(memory_path),
        create_sandbox(settings.sandbox_backend, settings.sandbox_image),
        model_backend,
        trace_dir=settings.trace_dir,
    )


_task_from_file = task_from_file


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
@click.option("--output", "-o", default="data/training_splits/sft_dataset.jsonl", help="Output JSONL file path")
@click.option("--memory-db", default="data/evoagent_memory.db", help="Path to SQLite memory file")
@click.option("--trace-dir", default=str(settings.trace_dir), help="Run traces (preferred source of full prompts)")
@click.option(
    "--exclude-held-out",
    multiple=True,
    default=("benchmarks/split.json", "benchmarks/quixbugs/split.json"),
    help="Benchmark split files whose held-out tasks are left out of the training data",
)
def export_dataset(output: str, memory_db: str, trace_dir: str, exclude_held_out: tuple) -> None:
    """Export verified repairs as an SFT dataset, plus DPO pairs built from traces."""
    curator = DatasetCurator(SQLiteMemoryStore(Path(memory_db)))
    held_out = curator.held_out_task_ids(Path(f) for f in exclude_held_out if f)
    out_path = Path(output)
    rows = curator.export_dataset_jsonl(out_path, trace_dir=Path(trace_dir), exclude_tasks=held_out)
    dpo_path = out_path.with_name(f"{out_path.stem}_dpo.jsonl")
    pairs = curator.export_dpo_jsonl(dpo_path, Path(trace_dir), exclude_tasks=held_out)
    manifest = curator.write_manifest(out_path.with_name("manifest.json"), [out_path, dpo_path], held_out)
    console.print(f"[bold green]Exported {rows} SFT rows to {out_path} and {pairs} DPO pairs to {dpo_path}[/]")
    console.print(f"Left out {len(held_out)} benchmark held-out task ids so they stay unseen.")
    splits = manifest["files"].get(out_path.name, {}).get("tasks_per_split", {})
    console.print(f"SFT tasks per split: {splits} (manifest: {out_path.with_name('manifest.json')})")


@main.command()
@click.option("--dataset", default="data/training_splits/sft_dataset.jsonl", help="SFT JSONL from export-dataset")
@click.option("--base-model", default=None, help="Hugging Face model id (default: Qwen2.5-Coder-1.5B-Instruct)")
@click.option("--output-dir", default="model_weights/evoagent_adapter", help="Where the adapter is saved")
@click.option("--epochs", default=3.0, type=float)
@click.option("--lr", default=2e-4, type=float)
@click.option("--max-seq-length", default=1536, help="Longest example in tokens; longer ones are skipped")
@click.option("--lora-r", default=16)
@click.option("--grad-accum", default=8, help="Gradient accumulation steps (batch size is 1)")
@click.option("--precision", type=click.Choice(["fp16", "bf16", "fp32"]), default="fp16",
              help="Compute precision; GTX 16xx/RTX 20xx support fp16, not bf16")
@click.option("--no-4bit", is_flag=True, help="Skip 4-bit quantization (CPU smoke runs, or GPUs with spare VRAM)")
def train_adapter(
    dataset: str, base_model: str, output_dir: str, epochs: float, lr: float, max_seq_length: int,
    lora_r: int, grad_accum: int, precision: str, no_4bit: bool,
) -> None:
    """Train a Q-LoRA adapter on exported SFT data (sized for a 4 GB GTX 1650)."""
    from evoagent.learning.finetune import QLoRAConfig, TrainingDataError
    from evoagent.learning.finetune import train_adapter as run_training

    overrides = {"base_model": base_model} if base_model else {}
    config = QLoRAConfig(
        dataset_path=Path(dataset), output_dir=Path(output_dir), epochs=epochs, learning_rate=lr,
        max_seq_length=max_seq_length, lora_r=lora_r, lora_alpha=2 * lora_r,
        gradient_accumulation_steps=grad_accum, precision=precision, quantize_4bit=not no_4bit, **overrides,
    )
    # Messages contain brackets (pip extras, "[warning]") that rich would read as markup
    try:
        summary = run_training(config, log=lambda message: console.print(message, markup=False))
    except (TrainingDataError, RuntimeError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise SystemExit(1)
    console.print(
        f"[bold green]Adapter trained on {summary['train_examples']} examples[/] "
        f"(train loss {summary['train_loss']:.4f}, val loss {summary['eval_loss']})."
    )


@main.command()
@click.option("--adapter", default="model_weights/evoagent_adapter", help="Adapter directory from train-adapter")
@click.option("--output-dir", default="model_weights/evoagent_merged", help="Where the merged model is saved")
@click.option("--base-model", default=None, help="Override the base model recorded in the adapter")
def merge_adapter(adapter: str, output_dir: str, base_model: str) -> None:
    """Merge an adapter into its base model so it can be converted to GGUF and served."""
    from evoagent.learning.finetune import merge_adapter as run_merge

    out = run_merge(Path(adapter), Path(output_dir), base_model)
    console.print(f"[bold green]Merged model saved to {out}[/]")
    console.print(
        "Next: python llama.cpp/convert_hf_to_gguf.py "
        f"{out} --outfile {out}.gguf --outtype q8_0, then serve it with llama-server or Ollama "
        "and re-run the held-out ablation against it."
    )


@main.command()
@click.option("--host", default="127.0.0.1", help="Interface to bind (0.0.0.0 inside containers)")
@click.option("--port", "-p", default=8000, help="Port to serve the API on")
@click.option("--embedded-worker", is_flag=True, help="Also run a worker in this process (single-machine setup)")
@click.option("--insecure-no-auth", is_flag=True, help="Accept requests without a bearer token (local development only)")
def serve(host: str, port: int, embedded_worker: bool, insecure_no_auth: bool) -> None:
    """Run the FastAPI job API (Milestone 5)."""
    import threading
    import uvicorn
    from evoagent.service.api import create_app
    from evoagent.service.runtime import Runtime

    runtime = Runtime(settings)
    job_store, queue, memory_store = runtime.job_store(), runtime.queue(), runtime.memory_store()
    if not embedded_worker and settings.queue_backend == "memory":
        console.print("[yellow]queue_backend=memory: jobs only run with --embedded-worker in this process.[/]")
    app = create_app(
        job_store, queue, memory_store,
        api_tokens=settings.split_list(settings.api_tokens),
        max_upload_bytes=settings.max_upload_bytes,
        allow_unauthenticated=insecure_no_auth,
    )
    stop = threading.Event()
    if embedded_worker:
        embedded = runtime.worker(queue, job_store, memory_store)
        threading.Thread(target=embedded.run_forever, args=(stop,), daemon=True).start()
        console.print(f"Embedded worker {embedded.worker_id} started")
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        stop.set()


@main.command()
@click.option("--worker-id", default=None, help="Consumer name in the queue (default: hostname + random suffix)")
@click.option("--once", is_flag=True, help="Process at most one job, then exit")
def worker(worker_id: str, once: bool) -> None:
    """Run a remote repair worker that takes jobs from Redis (Milestone 5)."""
    import logging
    import signal
    import threading
    from evoagent.service.runtime import Runtime

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    runtime = Runtime(settings)
    repair_worker = runtime.worker(runtime.queue(), runtime.job_store(), runtime.memory_store(), worker_id)
    console.print(Panel(
        f"[bold green]Worker {repair_worker.worker_id}[/]\n"
        f"Queue: {settings.queue_backend} | Jobs: {settings.job_store_backend} | Memory: {settings.memory_backend} | "
        f"Sandbox: {settings.sandbox_backend} | Model: {settings.model_provider}/{settings.model_name}"
    ))
    if once:
        status = repair_worker.run_once(block_seconds=30)
        console.print(f"Job finished: {status.value}" if status else "No job was waiting.")
        return
    stop = threading.Event()
    # SIGTERM (docker stop, Kubernetes) lets the job in progress finish before the worker exits
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    repair_worker.run_forever(stop)
    console.print("Worker stopped.")


@main.command()
@click.option("--output-dir", default=str(settings.trace_dir), help="Directory to write trace JSON files into")
@click.option("--limit", default=1000, help="Most recent finished jobs to export")
def export_traces(output_dir: str, limit: int) -> None:
    """Copy run traces stored by remote workers into a trace directory, for export-dataset."""
    from evoagent.service.runtime import Runtime

    store = Runtime(settings).job_store()
    if not hasattr(store, "traces"):
        console.print("[red]Traces are only stored by the postgres job store (EVOAGENT_JOB_STORE_BACKEND=postgres).[/]")
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = store.traces(limit=limit)
    for trace in traces:
        job_id = trace.pop("job_id")
        (out / f"{trace.get('task_id', 'task')}-job-{job_id}.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    console.print(f"[bold green]Wrote {len(traces)} traces to {out}[/]; next: evoagent export-dataset --trace-dir {out}")


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
