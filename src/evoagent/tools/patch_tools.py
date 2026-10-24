"""
Patch validation, safety checks, SEARCH/REPLACE block processor, and unified diff applicator.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple
from evoagent.domain.models import PatchProposal


class PatchTool:
    """
    Validates and applies patches using either:
    1. Search/Replace blocks (preferred for LLMs and small models: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE)
    2. Standard unified diffs
    """

    def __init__(self, root_path: Path, max_patch_lines: int = 250) -> None:
        self.root_path = root_path
        self.max_patch_lines = max_patch_lines

    def validate_patch(self, patch: PatchProposal) -> Tuple[bool, Optional[str]]:
        """
        Verify that a patch adheres to safety rules:
        - Target file is within repo boundary
        - Line count does not exceed max_patch_lines
        - Contains search/replace blocks or diff additions/deletions
        """
        target_path = (self.root_path / patch.target_file).resolve()
        if not target_path.is_relative_to(self.root_path.resolve()):
            return False, f"Path traversal rejected: {patch.target_file}"

        raw_text = patch.diff_unified.strip()
        lines = raw_text.splitlines()
        if not lines:
            return False, "Patch is empty"

        if len(lines) > self.max_patch_lines:
            return False, f"Patch exceeds maximum line limit ({len(lines)} > {self.max_patch_lines})"

        has_search_replace = "<<<<<<< SEARCH" in raw_text and ">>>>>>> REPLACE" in raw_text
        has_diff_changes = any(l.startswith(("+", "-")) and not l.startswith(("+++", "---")) for l in lines)

        if not (has_search_replace or has_diff_changes):
            return False, "Patch contains no valid SEARCH/REPLACE blocks or diff changes"

        return True, None

    def apply_unified_diff(self, patch: PatchProposal) -> Tuple[bool, str]:
        """
        Apply patch (Search/Replace blocks or Unified Diff) to the target file.
        Returns (success: bool, message: str).
        """
        is_valid, error_msg = self.validate_patch(patch)
        if not is_valid:
            return False, f"Validation failed: {error_msg}"

        target_file = self.root_path / patch.target_file
        if not target_file.exists():
            return False, f"Target file does not exist: {patch.target_file}"

        original_content = target_file.read_text(encoding="utf-8", errors="replace")

        # 1. Prefer Search/Replace block application if detected
        if "<<<<<<< SEARCH" in patch.diff_unified:
            success, new_content, err_msg = self._apply_search_replace_blocks(
                original_content, patch.diff_unified
            )
            if not success:
                return False, f"Search/Replace block application failed: {err_msg}"
        else:
            # 2. Fall back to unified diff hunks
            success, new_content = self._apply_hunks(original_content, patch.diff_unified)
            if not success:
                return False, "Failed to apply diff hunks to original file content"

        try:
            target_file.write_text(new_content, encoding="utf-8")
            return True, f"Successfully applied patch to {patch.target_file}"
        except Exception as exc:
            # Rollback
            target_file.write_text(original_content, encoding="utf-8")
            return False, f"Error writing patched content: {str(exc)}"

    def _apply_search_replace_blocks(
        self, original_text: str, patch_text: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Extract and apply <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks.
        Supports multiple consecutive blocks within a single file.
        """
        pattern = re.compile(
            r"<<<<<<<\s*SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>>\s*REPLACE",
            re.DOTALL,
        )
        matches = list(pattern.finditer(patch_text))
        if not matches:
            return False, original_text, "Malformed SEARCH/REPLACE syntax."

        current_text = original_text.replace("\r\n", "\n")

        for idx, match in enumerate(matches, 1):
            search_block = match.group(1).replace("\r\n", "\n")
            replace_block = match.group(2).replace("\r\n", "\n")

            if search_block in current_text:
                current_text = current_text.replace(search_block, replace_block, 1)
            else:
                # Flexible whitespace fallback: trim trailing whitespace per line
                s_trimmed = "\n".join(l.rstrip() for l in search_block.splitlines())
                c_trimmed = "\n".join(l.rstrip() for l in current_text.splitlines())
                if s_trimmed in c_trimmed:
                    # Reconstruct replacement
                    current_text = c_trimmed.replace(s_trimmed, replace_block, 1)
                else:
                    preview = search_block[:60] + "..." if len(search_block) > 60 else search_block
                    return False, original_text, f"SEARCH block #{idx} not found in target file: {preview!r}"

        return True, current_text, None

    def _apply_hunks(self, original_text: str, diff_text: str) -> Tuple[bool, str]:
        """
        Lightweight unified diff applicator for standard diff hunks.
        """
        diff_lines = diff_text.strip().splitlines()

        removals = []
        additions = []

        for line in diff_lines:
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                continue
            if line.startswith("-"):
                removals.append(line[1:])
            elif line.startswith("+"):
                additions.append(line[1:])

        orig_chunk = "\n".join(removals)
        new_chunk = "\n".join(additions)

        orig_norm = original_text.replace("\r\n", "\n")
        chunk_norm = orig_chunk.replace("\r\n", "\n")
        new_norm = new_chunk.replace("\r\n", "\n")

        if chunk_norm and chunk_norm in orig_norm:
            patched = orig_norm.replace(chunk_norm, new_norm, 1)
            return True, patched

        # Context-anchored insertion
        context_anchor = None
        for line in diff_lines:
            if line.startswith(("---", "+++", "@@")):
                continue
            if line.startswith(" ") or (line and not line.startswith(("-", "+"))):
                context_anchor = line[1:] if line.startswith(" ") else line
                break

        if not removals and additions and context_anchor:
            if context_anchor in orig_norm:
                patched = orig_norm.replace(context_anchor, f"{new_norm}\n{context_anchor}", 1)
                return True, patched

        return False, original_text
