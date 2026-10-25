"""
Convert the QuixBugs Python programs into EvoAgent benchmark tasks.

Usage:
    git clone --depth 1 https://github.com/jkoppel/QuixBugs /tmp/QuixBugs
    python benchmarks/import_quixbugs.py --source /tmp/QuixBugs

Each program becomes a task with the buggy source, its pytest file, and its JSON test data. A task
is kept only if its tests fail on the buggy version and pass on QuixBugs' correct version; the
correct versions are used for this check only and never written into a task. Every task gets a
defect class derived from the AST-normalized diff between the buggy and correct function, and the
split gives every held-out task a train task of the same class.
"""

import argparse
import ast
from collections import defaultdict
import difflib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from typing import Dict, List, Optional

COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!=", "is", "in"}
ARITHMETIC_OPS = {
    "+", "-", "*", "/", "//", "%", "**", "^", "&", "|", "<<", ">>",
    "+=", "-=", "*=", "/=", "//=", "%=", "^=", "&=", "|=",
}
BOOLEAN_OPS = {"and", "or", "not"}
CONDITIONAL_IMPORT_RE = re.compile(
    r"if pytest\.use_correct:\n\s+from correct_python_programs\.(\w+) import ([\w, ]+)\n"
    r"else:\n\s+from python_programs\.\w+ import [\w, ]+\n"
)
LOAD_TESTDATA = '''import json
from pathlib import Path


def load_json_testcases(algorithm):
    path = Path(__file__).parent / f"{algorithm}.json"
    with open(path) as data_file:
        return [json.loads(line) for line in data_file]
'''


def _function_lines(source: str, name: str) -> Optional[List[str]]:
    """The first top-level definition of `name`, re-rendered so formatting differences vanish."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            # Drop docstrings and other bare string statements; keep everything else
            node.body = [
                stmt for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str))
            ] or node.body
            return ast.unparse(node).splitlines()
    return None


def _tokens(line: str) -> List[str]:
    try:
        return [tok.string for tok in tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline)
                if tok.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER, tokenize.INDENT)]
    except (tokenize.TokenError, IndentationError):
        return line.split()


def _has_call(tokens: List[str]) -> bool:
    return any(name.isidentifier() and paren == "(" for name, paren in zip(tokens, tokens[1:]))


def _classify_tokens(buggy: List[str], correct: List[str]) -> str:
    if buggy != correct and sorted(buggy) == sorted(correct):
        return "swapped-operands"
    ops = [op for op in difflib.SequenceMatcher(a=buggy, b=correct, autojunk=False).get_opcodes() if op[0] != "equal"]
    removed = [tok for _, i1, i2, _, _ in ops for tok in buggy[i1:i2]]
    added = [tok for _, _, _, j1, j2 in ops for tok in correct[j1:j2]]

    if len(ops) == 1 and ops[0][0] == "replace" and len(removed) == len(added) == 1:
        old, new = removed[0], added[0]
        if old in COMPARISON_OPS and new in COMPARISON_OPS:
            return "comparison-operator"
        if old in ARITHMETIC_OPS and new in ARITHMETIC_OPS:
            return "arithmetic-operator"
        if old in BOOLEAN_OPS and new in BOOLEAN_OPS:
            return "boolean-operator"
        if old.isdigit() and new.isdigit():
            return "off-by-one" if abs(int(old) - int(new)) == 1 else "wrong-constant"
        if old.isidentifier() and new.isidentifier():
            return "wrong-variable"

    # Parentheses added or removed around the change carry no meaning of their own
    core_removed = [tok for tok in removed if tok not in ("(", ")")]
    core_added = [tok for tok in added if tok not in ("(", ")")]
    if (not core_removed and core_added in (["+", "1"], ["-", "1"])) or (
        not core_added and core_removed in (["+", "1"], ["-", "1"])
    ):
        return "off-by-one"
    if not removed:
        return "missing-call" if _has_call(added) else "missing-expression"
    if not added:
        return "extra-call" if _has_call(removed) else "extra-expression"
    return "multiple-edits"


def classify_defect(buggy_source: str, correct_source: str, name: str) -> str:
    """Coarse defect class from the diff between the buggy and the correct function."""
    buggy, correct = _function_lines(buggy_source, name), _function_lines(correct_source, name)
    if buggy is None or correct is None:
        return "other"
    blocks = [op for op in difflib.SequenceMatcher(a=buggy, b=correct, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not blocks:
        return "other"
    classes = set()
    for tag, i1, i2, j1, j2 in blocks:
        if tag == "insert":
            classes.add("missing-line")
        elif tag == "delete":
            classes.add("extra-line")
        elif i2 - i1 == j2 - j1:
            classes.update(_classify_tokens(_tokens(b), _tokens(c)) for b, c in zip(buggy[i1:i2], correct[j1:j2]))
        else:
            classes.add("missing-line" if j2 - j1 > i2 - i1 else "extra-line")
    return classes.pop() if len(classes) == 1 else "multiple-edits"


def _run_tests(code_dir: Path, test_file: str, timeout: int) -> str:
    """'pass', 'fail' or 'timeout' for one pytest run, isolated from this repository's config."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test_file],
            cwd=code_dir, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    return "pass" if proc.returncode == 0 else "fail"


def build_task(source: Path, name: str, out_dir: Path) -> Optional[str]:
    """Write one task under out_dir; returns its defect class, or None with the reason printed."""
    test_src = source / "python_testcases" / f"test_{name}.py"
    if not test_src.exists():
        print(f"skip {name}: no pytest file")
        return None
    test_code = test_src.read_text(encoding="utf-8")
    test_code, replaced = CONDITIONAL_IMPORT_RE.subn(r"from \1 import \2\n", test_code)
    if not replaced:
        print(f"skip {name}: unexpected import layout")
        return None
    test_code = test_code.replace("pytest.run_slow", "False")

    code_dir = out_dir / "code"
    (code_dir / "tests").mkdir(parents=True)
    shutil.copy(source / "python_programs" / f"{name}.py", code_dir / f"{name}.py")
    (code_dir / "tests" / f"test_{name}.py").write_text(test_code, encoding="utf-8")
    json_cases = source / "json_testcases" / f"{name}.json"
    if json_cases.exists():
        shutil.copy(json_cases, code_dir / "tests" / f"{name}.json")
        (code_dir / "tests" / "load_testdata.py").write_text(LOAD_TESTDATA, encoding="utf-8")
    if "from node import" in test_code:
        shutil.copy(source / "python_testcases" / "node.py", code_dir / "tests" / "node.py")

    test_file = f"tests/test_{name}.py"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "code"
        shutil.copytree(code_dir, work)
        buggy_result = _run_tests(work, test_file, timeout=30)
        shutil.copy(source / "correct_python_programs" / f"{name}.py", work / f"{name}.py")
        correct_result = _run_tests(work, test_file, timeout=60)
    if buggy_result == "pass" or correct_result != "pass":
        print(f"skip {name}: buggy={buggy_result}, correct={correct_result}")
        shutil.rmtree(out_dir)
        return None

    defect_class = classify_defect(
        (source / "python_programs" / f"{name}.py").read_text(encoding="utf-8"),
        (source / "correct_python_programs" / f"{name}.py").read_text(encoding="utf-8"),
        name,
    )
    task = {
        "task_id": f"quixbugs-{name}",
        "repo_dir": "code",
        "issue_description": f"{name} gives wrong results on some inputs; tests/test_{name}.py fails",
        "targeted_test": test_file,
        "regression_test": test_file,
        "max_attempts": 3,
        "timeout_seconds": 30,
        "metadata": {"target_file": f"{name}.py", "defect_class": defect_class, "source": "QuixBugs"},
    }
    (out_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    print(f"ok   {name}: {defect_class} (buggy tests: {buggy_result})")
    return defect_class


def sibling_split(classes: Dict[str, str]) -> Dict[str, List[str]]:
    """Alternate each defect class between train and held-out; singleton classes stay in train."""
    by_class: Dict[str, List[str]] = defaultdict(list)
    for task_id, defect_class in sorted(classes.items()):
        by_class[defect_class].append(task_id)
    train, held_out = [], []
    for members in by_class.values():
        for i, task_id in enumerate(members):
            (held_out if i % 2 else train).append(task_id)
    return {"train_tasks": sorted(train), "held_out_tasks": sorted(held_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import QuixBugs Python programs as EvoAgent tasks")
    parser.add_argument("--source", required=True, help="Path to a QuixBugs checkout")
    parser.add_argument("--output", default=str(Path(__file__).parent / "quixbugs"), help="Benchmark root to write")
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    (output / "tasks").mkdir(parents=True)

    names = sorted(
        p.stem for p in (source / "python_programs").glob("*.py") if p.stem != "node" and not p.stem.endswith("_test")
    )
    classes = {}
    for name in names:
        defect_class = build_task(source, name, output / "tasks" / name)
        if defect_class:
            classes[f"quixbugs-{name}"] = defect_class

    split = sibling_split(classes)
    split["description"] = (
        "QuixBugs programs split so that every held-out task has a train task with the same "
        "derived metadata.defect_class."
    )
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    shutil.copy(source / "LICENSE", output / "LICENSE")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (output / "README.md").write_text(
        "# QuixBugs tasks\n\n"
        "Generated by `python benchmarks/import_quixbugs.py --source <QuixBugs checkout>` from\n"
        f"https://github.com/jkoppel/QuixBugs (commit `{commit or 'unknown'}`). The programs, tests and\n"
        "test data are QuixBugs' own (MIT license, see LICENSE); each task holds the buggy program,\n"
        "its pytest file and its JSON test cases. The correct versions were used only to check that\n"
        "each task's tests fail before and pass after the fix, and are not included.\n\n"
        "`metadata.defect_class` is derived from the diff between the buggy and the correct function\n"
        "(`classify_defect` in the importer). The split alternates each class between train and\n"
        "held-out, so every held-out task has a train task with the same defect class.\n",
        encoding="utf-8",
    )
    print(f"\n{len(classes)} of {len(names)} programs imported: "
          f"{len(split['train_tasks'])} train, {len(split['held_out_tasks'])} held-out")


if __name__ == "__main__":
    main()
