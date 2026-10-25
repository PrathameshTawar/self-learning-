"""
Positive control for the held-out ablation: when memory genuinely helps, the runner must show it.
"""

import importlib.util
import json
from pathlib import Path
import shutil
from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse
from tests.unit.test_state_machine import GOOD_FIX

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_benchmarks", REPO_ROOT / "benchmarks" / "run_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeedsMemoryModel(IModelBackend):
    """Fixes the train task on its own, but the held-out sibling only when memory supplies a lesson."""

    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        if prompt.task_id == "train-parser" or prompt.retrieved_lessons:
            return ModelResponse(hypothesis="Guard the empty list before indexing.", proposed_diff=GOOD_FIX)
        return ModelResponse(hypothesis="No idea.", proposed_diff="")


def _make_benchmark(root: Path) -> None:
    source = REPO_ROOT / "benchmarks" / "tasks" / "bug_001_parser"
    for task_id in ("train-parser", "held-out-parser"):
        task_dir = root / "tasks" / task_id
        shutil.copytree(source, task_dir)
        task_json = task_dir / "task.json"
        data = json.loads(task_json.read_text(encoding="utf-8"))
        data["task_id"] = task_id
        task_json.write_text(json.dumps(data), encoding="utf-8")
    (root / "split.json").write_text(
        json.dumps({"train_tasks": ["train-parser"], "held_out_tasks": ["held-out-parser"]}), encoding="utf-8"
    )


def test_ablation_detects_a_real_memory_effect(tmp_path: Path):
    runner = _load_runner()
    _make_benchmark(tmp_path / "bench")
    results_path = tmp_path / "ablation_results.json"

    runner.run_held_out_comparison(tmp_path / "bench", NeedsMemoryModel(), results_path, trials=2)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    rates = {cond["id"]: cond["resolution_rate_per_trial"] for cond in results["conditions"]}
    assert rates == {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [100.0, 100.0]}
    assert results["learned_episodes_per_trial"] == [1, 1]
    assert results["memory_effect_points"] == 100.0
    # Two discordant pairs, both favouring C
    assert results["paired_b_vs_c"] == {"solved_only_by_b": 0, "solved_only_by_c": 2, "mcnemar_p": 0.5}
