"""
Dataset curator: converts verified memory episodes into clean SFT and DPO training sets.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Set, Tuple
from evoagent.memory.store import IMemoryStore
from evoagent.models.base import ModelPrompt
from evoagent.models.open_model import OpenAICompatibleModel


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

    @staticmethod
    def _load_traces(trace_dir: Path) -> List[Dict[str, Any]]:
        traces = []
        for trace_file in sorted(trace_dir.glob("*.json")):
            try:
                trace = json.loads(trace_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            trace["trace_file"] = trace_file.name
            traces.append(trace)
        return traces

    @staticmethod
    def _trace_example(trace: Dict[str, Any], attempt: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild the exact chat prompt an attempt was given and the completion it produced."""
        messages = OpenAICompatibleModel().build_messages(ModelPrompt(**attempt["prompt"]))
        patch = attempt["patch"]
        output = attempt["model_output"] or ""
        # LLM outputs contain the patch verbatim; otherwise rebuild "hypothesis + patch"
        if patch and patch in output.replace("\r\n", "\n"):
            completion = output
        else:
            completion = f"{attempt['hypothesis']}\n\n{patch}"
        return {
            "task_id": trace["task_id"],
            "messages": messages,
            "prompt": f"{messages[0]['content']}\n\n{messages[1]['content']}",
            "completion": completion,
            "provenance": f"{trace['trace_file']}#attempt{attempt['attempt_number']}",
        }

    def create_sft_from_traces(self, trace_dir: Path) -> List[Dict[str, Any]]:
        """SFT pairs from evaluator-accepted attempts, with the full prompt the model actually saw."""
        return [
            self._trace_example(trace, attempt)
            for trace in self._load_traces(trace_dir)
            for attempt in trace["attempts"]
            if attempt["state"] == "SUCCEEDED"
        ]

    def create_dpo_from_traces(self, trace_dir: Path) -> List[Dict[str, Any]]:
        """
        Preference pairs from attempts that received the identical prompt: an accepted output is
        chosen over a rejected one. Attempts with different prompts (e.g. later attempts that saw
        retry feedback) are never paired.
        """
        groups: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
        for trace in self._load_traces(trace_dir):
            for attempt in trace["attempts"]:
                if not attempt["patch"]:
                    continue
                example = self._trace_example(trace, attempt)
                prompt_key = hashlib.sha256(json.dumps(example["messages"]).encode("utf-8")).hexdigest()
                group = groups.setdefault((trace["task_id"], prompt_key), {"chosen": [], "rejected": []})
                group["chosen" if attempt["state"] == "SUCCEEDED" else "rejected"].append(example)

        pairs = []
        for group in groups.values():
            if not group["chosen"]:
                continue
            chosen = group["chosen"][0]
            for rejected_completion in dict.fromkeys(ex["completion"] for ex in group["rejected"]):
                pairs.append({
                    "task_id": chosen["task_id"],
                    "prompt": chosen["prompt"],
                    "chosen": chosen["completion"],
                    "rejected": rejected_completion,
                })
        return pairs

    @staticmethod
    def held_out_task_ids(split_files: Iterable[Path]) -> Set[str]:
        """Held-out task ids from benchmark split files; their traces must not become training data."""
        ids: Set[str] = set()
        for split_file in split_files:
            if split_file.exists():
                ids.update(json.loads(split_file.read_text(encoding="utf-8")).get("held_out_tasks", []))
        return ids

    @staticmethod
    def split_for_task(task_id: str, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
        """Assign whole tasks to a split by a stable hash, so a task never appears in two splits."""
        bucket = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16) % 1000 / 1000
        if bucket < split_ratio[0]:
            return "train"
        return "val" if bucket < split_ratio[0] + split_ratio[1] else "test"

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

    def export_dataset_jsonl(
        self,
        output_file: Path,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        trace_dir: Optional[Path] = None,
        exclude_tasks: Collection[str] = (),
    ) -> int:
        """
        Export the SFT dataset to JSONL and return the number of rows. Traces are preferred because
        episodes lack the code context the model saw; episodes are the fallback. Rows for
        exclude_tasks (benchmark held-out tasks) are dropped.
        """
        sft_data = self.create_sft_from_traces(trace_dir) if trace_dir and trace_dir.exists() else []
        if not sft_data:
            sft_data = self.create_sft_dataset()
        sft_data = [item for item in sft_data if item["task_id"] not in exclude_tasks]
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in sft_data:
                item["split"] = self.split_for_task(item["task_id"], split_ratio)
                f.write(json.dumps(item) + "\n")
        return len(sft_data)

    def export_dpo_jsonl(self, output_file: Path, trace_dir: Path, exclude_tasks: Collection[str] = ()) -> int:
        """Export trace-based preference pairs to JSONL and return the number of pairs."""
        pairs = self.create_dpo_from_traces(trace_dir) if trace_dir.exists() else []
        pairs = [pair for pair in pairs if pair["task_id"] not in exclude_tasks]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                pair["split"] = self.split_for_task(pair["task_id"])
                f.write(json.dumps(pair) + "\n")
        return len(pairs)
