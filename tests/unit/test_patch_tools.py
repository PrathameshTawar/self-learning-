"""
Unit tests for PatchTool validation, hunk application, and path security.
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
