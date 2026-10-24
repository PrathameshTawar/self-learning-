"""
Patch validation, safety checks, and unified diff applicator.
"""

from pathlib import Path
from typing import Optional, Tuple
from evoagent.domain.models import PatchProposal


class PatchTool:
    """Validates and applies unified diffs or safe code replacements."""

    def __init__(self, root_path: Path, max_patch_lines: int = 150) -> None:
        self.root_path = root_path
        self.max_patch_lines = max_patch_lines

    def validate_patch(self, patch: PatchProposal) -> Tuple[bool, Optional[str]]:
        """
        Verify that a patch adheres to safety rules:
        - Target file is within repo boundary
        - Line count does not exceed max_patch_lines
        - Diff contains modifications
        """
        target_path = (self.root_path / patch.target_file).resolve()
        if not target_path.is_relative_to(self.root_path.resolve()):
            return False, f"Path traversal rejected: {patch.target_file}"

        lines = patch.diff_unified.strip().splitlines()
        if not lines:
            return False, "Patch is empty"

        if len(lines) > self.max_patch_lines:
            return False, f"Patch exceeds maximum line limit ({len(lines)} > {self.max_patch_lines})"

        has_changes = any(l.startswith(("+", "-")) and not l.startswith(("+++", "---")) for l in lines)
        if not has_changes:
            return False, "Patch diff contains no additions or deletions"

        return True, None

    def apply_unified_diff(self, patch: PatchProposal) -> Tuple[bool, str]:
        """
        Apply unified diff to the target file.
        Returns (success: bool, message: str).
        """
        is_valid, error_msg = self.validate_patch(patch)
        if not is_valid:
            return False, f"Validation failed: {error_msg}"

        target_file = self.root_path / patch.target_file
        if not target_file.exists():
            return False, f"Target file does not exist: {patch.target_file}"

        original_content = target_file.read_text(encoding="utf-8", errors="replace")

        # Parse simple unified diff hunks
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

    def _apply_hunks(self, original_text: str, diff_text: str) -> Tuple[bool, str]:
        """
        Lightweight unified diff applicator that parses standard diff hunks.
        Falls back to exact chunk replacement if diff lines match.
        """
        diff_lines = diff_text.strip().splitlines()

        # Find hunk headers @@ -orig,len +new,len @@
        removals = []
        additions = []

        for line in diff_lines:
            if line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("@@"):
                continue
            if line.startswith("-"):
                removals.append(line[1:])
            elif line.startswith("+"):
                additions.append(line[1:])

        orig_chunk = "\n".join(removals)
        new_chunk = "\n".join(additions)

        # Normalize line endings
        orig_norm = original_text.replace("\r\n", "\n")
        chunk_norm = orig_chunk.replace("\r\n", "\n")
        new_norm = new_chunk.replace("\r\n", "\n")

        if chunk_norm and chunk_norm in orig_norm:
            patched = orig_norm.replace(chunk_norm, new_norm, 1)
            return True, patched

        # Context-anchored insertion (e.g. inserting guard before existing statement)
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
