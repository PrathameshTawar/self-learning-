"""
Unit tests for Layer C dataset curation from run traces.
"""

import json
from pathlib import Path
from evoagent.domain.models import ExecutionBudget, Task, TestResult
from evoagent.learning.dataset import DatasetCurator
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore
from evoagent.orchestration.planner import TaskPlanner
from tests.unit.test_state_machine import GOOD_FIX, WRONG_FIX, RecordingModel, _parser_task, _real_state_machine


def _run(repo: Path, trace_dir: Path, diffs, attempts: int = 3) -> None:
    model = RecordingModel(diffs)
    sm = _real_state_machine(repo, model, InMemoryMemoryStore(), trace_dir)
    sm.execute_task(_parser_task(repo, attempts=attempts), persist_episode=False)


def test_failure_text_removes_per_run_noise(tmp_path: Path):
    task = Task(
        task_id="t", repository_path=tmp_path / "work", issue_description="", targeted_test="tests",
        budget=ExecutionBudget(),
    )

    def result(stdout: str) -> TestResult:
        return TestResult(command="pytest", passed=False, exit_code=1, stdout=stdout, stderr="", duration_seconds=0.1)

    first = TaskPlanner.failure_text(task, result(f"E   IndexError\n{tmp_path / 'work'}/a.py\n1 failed in 0.01s"))
    second = TaskPlanner.failure_text(task, result(f"E   IndexError\n{tmp_path / 'work'}/a.py\n1 failed in 3.27s"))
    assert first == second == "E   IndexError\n./a.py\n1 failed\n"


def test_sft_examples_carry_the_full_prompt_the_model_saw(temp_repo: Path, tmp_path: Path):
    trace_dir = tmp_path / "traces"
    _run(temp_repo, trace_dir, [WRONG_FIX, GOOD_FIX])

    examples = DatasetCurator(InMemoryMemoryStore()).create_sft_from_traces(trace_dir)

    assert len(examples) == 1
    example = examples[0]
    # Episode-based prompts only had the issue text; trace prompts include code and feedback
    assert "def first_field" in example["prompt"]
    assert "PREVIOUS FAILED ATTEMPT" in example["prompt"]
    assert example["completion"].endswith(GOOD_FIX)
    assert example["messages"][0]["role"] == "system"


def test_dpo_pairs_only_outputs_for_identical_prompts(temp_repo: Path, tmp_path: Path):
    trace_dir = tmp_path / "traces"
    _run(temp_repo, trace_dir, [GOOD_FIX], attempts=1)    # attempt 1 accepted
    _run(temp_repo, trace_dir, [WRONG_FIX], attempts=1)   # same first prompt, rejected
    _run(temp_repo, trace_dir, [WRONG_FIX, GOOD_FIX])     # second prompt differs (has feedback)

    pairs = DatasetCurator(InMemoryMemoryStore()).create_dpo_from_traces(trace_dir)

    assert len(pairs) == 1
    assert pairs[0]["chosen"].endswith(GOOD_FIX)
    assert pairs[0]["rejected"].endswith(WRONG_FIX)
    assert "PREVIOUS FAILED ATTEMPT" not in pairs[0]["prompt"]


def test_export_keeps_each_task_in_a_single_split(tmp_path: Path):
    store = InMemoryMemoryStore()
    curator = DatasetCurator(store)
    out = tmp_path / "sft.jsonl"
    rows = [{"task_id": f"task-{i % 3}", "prompt": "p", "completion": "c"} for i in range(30)]
    curator.create_sft_dataset = lambda: rows

    assert curator.export_dataset_jsonl(out) == 30
    splits_per_task = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits_per_task.setdefault(row["task_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_per_task.values())


def test_export_leaves_out_benchmark_held_out_tasks(temp_repo: Path, tmp_path: Path):
    trace_dir = tmp_path / "traces"
    _run(temp_repo, trace_dir, [GOOD_FIX], attempts=1)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train_tasks": [], "held_out_tasks": ["parser-empty-input"]}), encoding="utf-8")
    curator = DatasetCurator(InMemoryMemoryStore())

    held_out = curator.held_out_task_ids([split_file, tmp_path / "missing.json"])
    assert held_out == {"parser-empty-input"}
    assert curator.export_dataset_jsonl(tmp_path / "sft.jsonl", trace_dir=trace_dir) == 1
    assert curator.export_dataset_jsonl(tmp_path / "sft.jsonl", trace_dir=trace_dir, exclude_tasks=held_out) == 0
