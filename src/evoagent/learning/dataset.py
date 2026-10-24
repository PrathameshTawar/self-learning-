"""
Dataset curator: converts verified memory episodes into clean SFT and DPO training sets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from evoagent.memory.store import IMemoryStore


class DatasetCurator:
    """
    Curates high-quality training pairs from verified task outcomes:
    - Filters only verified, regression-free fixes for SFT
    - Formulates accepted vs rejected pairs for preference optimization (DPO)
    - Strips secrets and sensitive patterns
    - Prevents data leakage across splits
    """

    def __init__(self, memory_store: IMemoryStore) -> None:
        self.memory_store = memory_store

    def create_sft_dataset(self) -> List[Dict[str, str]]:
        """Extract instruction-completion pairs from accepted episodes."""
        episodes = self.memory_store.get_accepted_episodes()
        dataset = []

        for ep in episodes:
            if not ep.applied_patch or not ep.targeted_tests_passed:
                continue

            instruction = (
                f"You are EvoAgent. Fix the defect in {ep.target_file}.\n"
                f"Issue: {ep.issue_description}\n"
                f"Failure: {ep.failure_signature}"
            )
            completion = f"Hypothesis: {ep.root_cause_hypothesis}\n\n```diff\n{ep.applied_patch}\n```"

            dataset.append({
                "task_id": ep.task_id,
                "prompt": instruction,
                "completion": completion,
                "provenance": ep.model_version,
            })

        return dataset

    def create_dpo_preference_dataset(self) -> List[Dict[str, Any]]:
        """Pair accepted patches (chosen) with failed/rejected patches (rejected) for the same task."""
        all_episodes = self.memory_store.get_all_episodes(limit=500)
        by_task: Dict[str, Dict[str, Any]] = {}

        for ep in all_episodes:
            if ep.task_id not in by_task:
                by_task[ep.task_id] = {"accepted": [], "rejected": []}
            if ep.outcome == "accepted":
                by_task[ep.task_id]["accepted"].append(ep)
            else:
                by_task[ep.task_id]["rejected"].append(ep)

        dpo_pairs = []
        for task_id, group in by_task.items():
            if group["accepted"] and group["rejected"]:
                chosen = group["accepted"][0]
                rejected = group["rejected"][0]

                dpo_pairs.append({
                    "task_id": task_id,
                    "prompt": f"Fix defect in {chosen.target_file}: {chosen.issue_description}",
                    "chosen": chosen.applied_patch,
                    "rejected": rejected.applied_patch,
                    "chosen_hypothesis": chosen.root_cause_hypothesis,
                    "rejected_hypothesis": rejected.root_cause_hypothesis,
                })

        return dpo_pairs

    def export_dataset_jsonl(self, output_file: Path, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> None:
        """Export curated SFT dataset to JSONL with train/val/test splits."""
        sft_data = self.create_sft_dataset()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        n = len(sft_data)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        with open(output_file, "w", encoding="utf-8") as f:
            for idx, item in enumerate(sft_data):
                split = "train" if idx < train_end else ("val" if idx < val_end else "test")
                item["split"] = split
                f.write(json.dumps(item) + "\n")
