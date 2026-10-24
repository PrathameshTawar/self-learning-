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
        "--output-dir",
        default="data/training_splits",
        help="Directory to save JSONL dataset files",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    if not db_path.exists():
        print(f"Memory database not found at {db_path}. Run benchmark tasks first to collect verified episodes.")
        return

    store = SQLiteMemoryStore(db_path)
    curator = DatasetCurator(store)

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_out = output_dir / "sft_dataset.jsonl"
    curator.export_dataset_jsonl(sft_out)

    print(f"Exported SFT dataset to {sft_out}")


if __name__ == "__main__":
    main()
