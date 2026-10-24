"""
Patch validation, safety checks, SEARCH/REPLACE block processor, and unified diff applicator.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple
from evoagent.domain.models import PatchProposal

# One SEARCH/REPLACE block. Either section may be empty; the divider and closing marker must start a line.
SEARCH_REPLACE_BLOCK_RE = re.compile(
    r"<<<<<<<[ \t]*SEARCH[ \t]*\r?\n(.*?)^=======[ \t]*\r?\n(.*?)^>>>>>>>[ \t]*REPLACE",
    re.DOTALL | re.MULTILINE,
)
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)")


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
            success, new_content, err_msg = self._apply_hunks(original_content, patch.diff_unified)
            if not success:
                return False, f"Failed to apply diff hunks: {err_msg}"

        try:
            target_file.write_text(new_content, encoding="utf-8")
            return True, f"Successfully applied patch to {patch.target_file}"
        except Exception as exc:
            # Rollback
            target_file.write_text(original_content, encoding="utf-8")
            return False, f"Error writing patched content: {str(exc)}"

    @staticmethod
    def _split_block(block: str) -> List[str]:
        """Split a SEARCH/REPLACE section into lines, dropping the newline that ends the section."""
        if block.endswith("\n"):
            block = block[:-1]
        return block.split("\n") if block else []

    @staticmethod
    def _find_block(
        lines: List[str], block: List[str], hint: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Locate block as consecutive whole lines: exact match first, then ignoring trailing whitespace.
        Returns (start_index, None) or (None, error). Several matches are ambiguous unless a 0-based
        line hint is given, in which case the closest match wins.
        """
        if not block:
            return None, "has no lines to anchor it"

        size = len(block)
        for normalize in (None, str.rstrip):
            haystack = lines if normalize is None else [normalize(l) for l in lines]
            needle = block if normalize is None else [normalize(l) for l in block]
            matches = [i for i in range(len(haystack) - size + 1) if haystack[i:i + size] == needle]
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                if hint is not None:
                    return min(matches, key=lambda i: abs(i - hint)), None
                return None, f"matches {len(matches)} locations; include more surrounding lines to make it unique"

        return None, "was not found in target file"

    def _apply_search_replace_blocks(
        self, original_text: str, patch_text: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Apply <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks in order.
        Each SEARCH section must match whole lines exactly once; an empty REPLACE deletes the lines.
        """
        matches = list(SEARCH_REPLACE_BLOCK_RE.finditer(patch_text.replace("\r\n", "\n")))
        if not matches:
            return False, original_text, "Malformed SEARCH/REPLACE syntax."

        lines = original_text.replace("\r\n", "\n").split("\n")
        for idx, match in enumerate(matches, 1):
            search_lines = self._split_block(match.group(1))
            replace_lines = self._split_block(match.group(2))
            start, error = self._find_block(lines, search_lines)
            if start is None:
                preview = "\n".join(search_lines)
                preview = preview[:60] + "..." if len(preview) > 60 else preview
                return False, original_text, f"SEARCH block #{idx} {error}: {preview!r}"
            lines[start:start + len(search_lines)] = replace_lines

        return True, "\n".join(lines), None

    @staticmethod
    def _parse_hunks(diff_text: str) -> List[Tuple[Optional[int], List[str], List[str]]]:
        """Split a unified diff into (0-based line hint, before_lines, after_lines) per hunk."""
        hunks: List[Tuple[Optional[int], List[str], List[str]]] = []
        current: Optional[Tuple[Optional[int], List[str], List[str]]] = None

        for line in diff_text.replace("\r\n", "\n").rstrip("\n").split("\n"):
            if line.startswith("@@"):
                header = HUNK_HEADER_RE.match(line)
                current = (int(header.group(1)) - 1 if header else None, [], [])
                hunks.append(current)
                continue
            if current is None:
                # File headers and preamble before the first hunk
                if not line or line.startswith(("--- ", "+++ ", "diff ", "index ")) or line in ("---", "+++"):
                    continue
                current = (None, [], [])
                hunks.append(current)
            if line.startswith("\\"):
                continue  # "\ No newline at end of file"
            if line.startswith("-"):
                current[1].append(line[1:])
            elif line.startswith("+"):
                current[2].append(line[1:])
            else:
                context = line[1:] if line.startswith(" ") else line
                current[1].append(context)
                current[2].append(context)

        return [hunk for hunk in hunks if hunk[1] or hunk[2]]

    def _apply_hunks(self, original_text: str, diff_text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Apply unified diff hunks by content: each hunk's context and removed lines must match whole
        lines in the file. Line numbers from @@ headers only break ties, so stale numbers are tolerated.
        """
        hunks = self._parse_hunks(diff_text)
        if not hunks:
            return False, original_text, "no hunks found in diff"

        lines = original_text.replace("\r\n", "\n").split("\n")
        offset = 0
        for idx, (hint, before, after) in enumerate(hunks, 1):
            start, error = self._find_block(lines, before, None if hint is None else hint + offset)
            if start is None:
                return False, original_text, f"hunk #{idx} {error}"
            lines[start:start + len(before)] = after
            offset += len(after) - len(before)

        return True, "\n".join(lines), None
