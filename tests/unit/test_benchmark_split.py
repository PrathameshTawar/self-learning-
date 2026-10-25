"""
Guards for the bundled benchmark split used by the held-out ablation.
"""

import json
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _tasks_by_id():
    tasks = {}
    for task_file in (BENCHMARKS / "tasks").glob("*/task.json"):
        data = json.loads(task_file.read_text(encoding="utf-8"))
        tasks[data["task_id"]] = data
    return tasks


def test_split_is_disjoint_and_complete():
    split = json.loads((BENCHMARKS / "split.json").read_text(encoding="utf-8"))
    train, held_out = set(split["train_tasks"]), set(split["held_out_tasks"])

    assert not train & held_out
    assert train | held_out == set(_tasks_by_id())


def test_every_held_out_task_has_a_train_sibling_with_the_same_defect_class():
    split = json.loads((BENCHMARKS / "split.json").read_text(encoding="utf-8"))
    tasks = _tasks_by_id()
    train_classes = {tasks[tid]["metadata"]["defect_class"] for tid in split["train_tasks"]}

    for tid in split["held_out_tasks"]:
        assert tasks[tid]["metadata"]["defect_class"] in train_classes, tid
