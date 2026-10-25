"""
Guards for the bundled benchmark split used by the held-out ablation.
"""

import importlib.util
import json
from pathlib import Path
import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
ROOTS = [BENCHMARKS, BENCHMARKS / "quixbugs"]


def _tasks_by_id(root: Path):
    tasks = {}
    for task_file in (root / "tasks").glob("*/task.json"):
        data = json.loads(task_file.read_text(encoding="utf-8"))
        tasks[data["task_id"]] = data
    return tasks


@pytest.mark.parametrize("root", ROOTS, ids=lambda r: r.name)
def test_split_is_disjoint_and_complete(root: Path):
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    train, held_out = set(split["train_tasks"]), set(split["held_out_tasks"])

    assert not train & held_out
    assert train | held_out == set(_tasks_by_id(root))


@pytest.mark.parametrize("root", ROOTS, ids=lambda r: r.name)
def test_every_held_out_task_has_a_train_sibling_with_the_same_defect_class(root: Path):
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    tasks = _tasks_by_id(root)
    train_classes = {tasks[tid]["metadata"]["defect_class"] for tid in split["train_tasks"]}

    for tid in split["held_out_tasks"]:
        assert tasks[tid]["metadata"]["defect_class"] in train_classes, tid


def test_quixbugs_tasks_ship_without_correct_versions():
    for task_dir in (BENCHMARKS / "quixbugs" / "tasks").iterdir():
        files = {p.name for p in (task_dir / "code").rglob("*") if p.is_file()}
        assert not any("correct" in name for name in files), task_dir.name


def _importer():
    spec = importlib.util.spec_from_file_location("import_quixbugs", BENCHMARKS / "import_quixbugs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "buggy, correct, expected",
    [
        ("def f(a, b):\n    return a < b\n", "def f(a, b):\n    return a <= b\n", "comparison-operator"),
        ("def f(a, b):\n    return g(a % b, b)\n", "def f(a, b):\n    return g(b, a % b)\n", "swapped-operands"),
        ("def f(x):\n    y = x + 1\n    return y\n", "def f(x):\n    y = max(0, x + 1)\n    return y\n", "missing-call"),
        ("def f(xs):\n    return xs[: len(xs)]\n", "def f(xs):\n    return xs[: len(xs) - 1]\n", "off-by-one"),
        ("def f(xs):\n    out = []\n    return out\n", "def f(xs):\n    out = []\n    out.append(1)\n    return out\n", "missing-line"),
        # The docstring on one side only must not count as an edit
        ('def f(a, b):\n    """Doc."""\n    return 0\n', "def f(a, b):\n    return 5\n", "wrong-constant"),
    ],
)
def test_defect_classifier(buggy: str, correct: str, expected: str):
    assert _importer().classify_defect(buggy, correct, "f") == expected
