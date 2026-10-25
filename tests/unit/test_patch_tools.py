"""
Unit tests for PatchTool validation, hunk application, SEARCH/REPLACE blocks, and path security.
"""

from pathlib import Path
from evoagent.domain.models import PatchProposal
from evoagent.tools.patch_tools import PatchTool


def test_validate_patch_rejects_path_traversal(tmp_path: Path):
    tool = PatchTool(root_path=tmp_path)
    malicious_patch = PatchProposal(
        target_file="../outside.py",
        diff_unified="--- a/outside.py\n+++ b/outside.py\n@@ -1 +1 @@\n-a\n+b",
    )
    is_valid, err = tool.validate_patch(malicious_patch)
    assert not is_valid
    assert "Path traversal rejected" in err


def test_validate_patch_rejects_empty_diff(tmp_path: Path):
    tool = PatchTool(root_path=tmp_path)
    empty_patch = PatchProposal(
        target_file="test.py",
        diff_unified="",
    )
    is_valid, err = tool.validate_patch(empty_patch)
    assert not is_valid
    assert "Patch is empty" in err


def test_validate_patch_rejects_exceeding_line_budget(tmp_path: Path):
    tool = PatchTool(root_path=tmp_path, max_patch_lines=5)
    long_diff = "\n".join(["+line"] * 10)
    patch = PatchProposal(
        target_file="test.py",
        diff_unified=f"--- a/test.py\n+++ b/test.py\n{long_diff}",
    )
    is_valid, err = tool.validate_patch(patch)
    assert not is_valid
    assert "exceeds maximum line limit" in err


def test_apply_unified_diff_success(tmp_path: Path):
    code_file = tmp_path / "hello.py"
    code_file.write_text("def msg():\n    return 'bad'\n", encoding="utf-8")

    tool = PatchTool(root_path=tmp_path)
    diff = (
        "--- a/hello.py\n"
        "+++ b/hello.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-    return 'bad'\n"
        "+    return 'good'"
    )
    patch = PatchProposal(target_file="hello.py", diff_unified=diff)

    success, msg = tool.apply_unified_diff(patch)
    assert success
    assert "good" in code_file.read_text(encoding="utf-8")


def test_apply_search_replace_block_success(tmp_path: Path):
    code_file = tmp_path / "service.py"
    code_file.write_text(
        "def compute(a, b):\n"
        "    return a / b\n",
        encoding="utf-8",
    )

    tool = PatchTool(root_path=tmp_path)
    sr_block = (
        "<<<<<<< SEARCH\n"
        "def compute(a, b):\n"
        "    return a / b\n"
        "=======\n"
        "def compute(a, b):\n"
        "    if b == 0:\n"
        "        return 0.0\n"
        "    return a / b\n"
        ">>>>>>> REPLACE"
    )
    patch = PatchProposal(target_file="service.py", diff_unified=sr_block)

    is_valid, err = tool.validate_patch(patch)
    assert is_valid, err

    success, msg = tool.apply_unified_diff(patch)
    assert success, msg
    updated_content = code_file.read_text(encoding="utf-8")
    assert "if b == 0:" in updated_content
    assert "return 0.0" in updated_content


def test_apply_search_replace_multiple_blocks(tmp_path: Path):
    code_file = tmp_path / "calc.py"
    code_file.write_text(
        "x = 10\n"
        "y = 20\n"
        "z = 30\n",
        encoding="utf-8",
    )

    tool = PatchTool(root_path=tmp_path)
    sr_blocks = (
        "<<<<<<< SEARCH\n"
        "x = 10\n"
        "=======\n"
        "x = 100\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "z = 30\n"
        "=======\n"
        "z = 300\n"
        ">>>>>>> REPLACE"
    )
    patch = PatchProposal(target_file="calc.py", diff_unified=sr_blocks)
    success, msg = tool.apply_unified_diff(patch)
    assert success, msg
    updated = code_file.read_text(encoding="utf-8")
    assert "x = 100" in updated
    assert "y = 20" in updated
    assert "z = 300" in updated


def _apply(tmp_path: Path, content: str, patch_text: str):
    code_file = tmp_path / "mod.py"
    code_file.write_text(content, encoding="utf-8")
    success, msg = PatchTool(root_path=tmp_path).apply_unified_diff(
        PatchProposal(target_file="mod.py", diff_unified=patch_text)
    )
    return success, msg, code_file.read_text(encoding="utf-8")


def test_search_replace_matches_whole_lines_only(tmp_path: Path):
    # "x = 1" is also a substring of "max_x = 1"; only the real line may change
    success, msg, updated = _apply(
        tmp_path,
        "max_x = 1\nx = 1\n",
        "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE",
    )
    assert success, msg
    assert updated == "max_x = 1\nx = 2\n"


def test_search_replace_rejects_ambiguous_match(tmp_path: Path):
    original = "def a():\n    return None\n\ndef b():\n    return None\n"
    success, msg, updated = _apply(
        tmp_path,
        original,
        "<<<<<<< SEARCH\n    return None\n=======\n    return 0\n>>>>>>> REPLACE",
    )
    assert not success
    assert "matches 2 locations" in msg
    assert updated == original


def test_search_replace_empty_replace_deletes_lines(tmp_path: Path):
    success, msg, updated = _apply(
        tmp_path,
        "a = 1\nprint('debug')\nb = 2\n",
        "<<<<<<< SEARCH\nprint('debug')\n=======\n>>>>>>> REPLACE",
    )
    assert success, msg
    assert updated == "a = 1\nb = 2\n"


def test_search_replace_tolerates_trailing_whitespace_and_keeps_final_newline(tmp_path: Path):
    success, msg, updated = _apply(
        tmp_path,
        "a = 1   \nb = 2\n",
        "<<<<<<< SEARCH\na = 1\n=======\na = 10\n>>>>>>> REPLACE",
    )
    assert success, msg
    assert updated == "a = 10\nb = 2\n"


def test_unified_diff_applies_multiple_hunks_with_context(tmp_path: Path):
    original = (
        "def low(xs):\n"
        "    return xs[0]\n"
        "\n"
        "def high(xs):\n"
        "    return xs[-1]\n"
    )
    diff = (
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def low(xs):\n"
        "-    return xs[0]\n"
        "+    return xs[0] if xs else None\n"
        "@@ -4,2 +4,2 @@\n"
        " def high(xs):\n"
        "-    return xs[-1]\n"
        "+    return xs[-1] if xs else None\n"
    )
    success, msg, updated = _apply(tmp_path, original, diff)
    assert success, msg
    assert updated == (
        "def low(xs):\n"
        "    return xs[0] if xs else None\n"
        "\n"
        "def high(xs):\n"
        "    return xs[-1] if xs else None\n"
    )


def test_unified_diff_uses_hunk_line_number_to_break_ties(tmp_path: Path):
    original = "total = 0\nstep()\ntotal = 0\nstep()\n"
    diff = "@@ -3,1 +3,1 @@\n-total = 0\n+total = 1\n"
    success, msg, updated = _apply(tmp_path, original, diff)
    assert success, msg
    assert updated == "total = 0\nstep()\ntotal = 1\nstep()\n"


def test_search_without_enclosing_indentation_is_reindented(tmp_path: Path):
    # Small models often copy code without the indentation of its enclosing function
    original = "def init(m):\n    dp = []\n    for i in range(m + 1):\n        dp.append(i)\n    return dp\n"
    success, msg, updated = _apply(
        tmp_path,
        original,
        "<<<<<<< SEARCH\nfor i in range(m + 1):\n    dp.append(i)\n=======\nfor i in range(m + 2):\n    dp.append(i)\n>>>>>>> REPLACE",
    )
    assert success, msg
    assert updated == "def init(m):\n    dp = []\n    for i in range(m + 2):\n        dp.append(i)\n    return dp\n"


def test_over_indented_search_is_shifted_back(tmp_path: Path):
    success, msg, updated = _apply(
        tmp_path,
        "x = 1\ny = 2\n",
        "<<<<<<< SEARCH\n    x = 1\n=======\n    x = 10\n>>>>>>> REPLACE",
    )
    assert success, msg
    assert updated == "x = 10\ny = 2\n"


def test_inconsistent_indentation_is_not_guessed(tmp_path: Path):
    original = "def f():\n    a = 1\n    b = 2\n"
    success, msg, updated = _apply(
        tmp_path,
        original,
        "<<<<<<< SEARCH\na = 1\n        b = 2\n=======\na = 3\n        b = 4\n>>>>>>> REPLACE",
    )
    assert not success
    assert "was not found" in msg
    assert updated == original
