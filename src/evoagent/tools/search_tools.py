"""
Read-only code inspection and symbol search tools.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


class CodeSearcher:
    """Safe read-only search tools for codebase inspection."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path

    def read_file(
        self,
        rel_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read content of a file with optional line boundaries."""
        target = (self.root_path / rel_path).resolve()
        if not target.is_relative_to(self.root_path.resolve()):
            raise PermissionError(f"Access outside repository path denied: {rel_path}")

        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {rel_path}")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line) if start_line else 1
        end = min(len(lines), end_line) if end_line else len(lines)

        numbered = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
        return "\n".join(numbered)

    def search_pattern(
        self,
        pattern: str,
        file_extension: str = ".py",
        max_matches: int = 25,
    ) -> List[Dict[str, str]]:
        """Search for a regex or string pattern across files in the repo."""
        compiled = re.compile(pattern)
        matches = []

        for root, _, files in os.walk(self.root_path):
            # Skip hidden and cache folders
            if any(part.startswith(".") or part in ("__pycache__", "venv") for part in Path(root).parts):
                continue

            for fname in files:
                if fname.endswith(file_extension):
                    file_path = Path(root) / fname
                    rel = file_path.relative_to(self.root_path).as_posix()
                    try:
                        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        for idx, line in enumerate(lines, 1):
                            if compiled.search(line):
                                matches.append({
                                    "file": rel,
                                    "line": str(idx),
                                    "content": line.strip(),
                                })
                                if len(matches) >= max_matches:
                                    return matches
                    except Exception:
                        continue

        return matches

    def list_files(self, extension: str = ".py") -> List[str]:
        """List all source files matching the extension."""
        found = []
        for root, _, files in os.walk(self.root_path):
            if any(part.startswith(".") or part in ("__pycache__", "venv") for part in Path(root).parts):
                continue
            for f in files:
                if f.endswith(extension):
                    found.append((Path(root) / f).relative_to(self.root_path).as_posix())
        return sorted(found)
