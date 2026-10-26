"""
Prepare verified EvoAgent episodic traces into Hugging Face / TRL SFT and DPO datasets.
"""

import argparse
from pathlib import Path
from evoagent.learning.dataset import DatasetCurator
from evoagent.memory.store import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EvoAgent training datasets")
    parser.add_argument(
        "--db-path",
        default="data/evoagent_memory.db",
        help="Path to SQLite memory database",
    )
    parser.add_argument(
        "--trace-dir",
        default="logs/traces",
        help="Run traces; preferred because they contain the full prompt the model saw",
    )
    parser.add_argument(
        "--exclude-held-out",
        nargs="*",
        default=["benchmarks/split.json", "benchmarks/quixbugs/split.json"],
        help="Benchmark split files whose held-out tasks are left out of the training data",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training_splits",
        help="Directory to save JSONL dataset files",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)

    if not db_path.exists() and not trace_dir.exists():
        print(f"Neither {db_path} nor {trace_dir} exists. Run benchmark tasks first to collect verified repairs.")
        return

    store = SQLiteMemoryStore(db_path)
    curator = DatasetCurator(store)

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_out = output_dir / "sft_dataset.jsonl"
    held_out = curator.held_out_task_ids(Path(f) for f in args.exclude_held_out)
    rows = curator.export_dataset_jsonl(sft_out, trace_dir=trace_dir, exclude_tasks=held_out)
    dpo_out = output_dir / "dpo_dataset.jsonl"
    pairs = curator.export_dpo_jsonl(dpo_out, trace_dir, exclude_tasks=held_out)
    curator.write_manifest(output_dir / "manifest.json", [sft_out, dpo_out], held_out)

    print(f"Exported {rows} SFT rows to {sft_out} and {pairs} DPO pairs to {dpo_out}")


if __name__ == "__main__":
    main()
