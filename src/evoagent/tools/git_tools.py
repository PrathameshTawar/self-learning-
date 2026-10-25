"""
Git inspection and repository snapshot management tool.
"""

from pathlib import Path
from typing import List, Optional
from evoagent.sandbox.interface import ISandboxRunner


class GitInspector:
    """Safe read-only and snapshot inspection operations on Git repositories."""

    def __init__(self, sandbox: ISandboxRunner) -> None:
        self.sandbox = sandbox

    def get_diff(self, repo_path: Path) -> str:
        """Get the current uncommitted git diff of the repository."""
        outcome = self.sandbox.execute_command("git diff", cwd=repo_path)
        return outcome.stdout if outcome.passed else ""

    def get_status(self, repo_path: Path) -> str:
        """Get git status output in short format."""
        outcome = self.sandbox.execute_command("git status --short", cwd=repo_path)
        return outcome.stdout if outcome.passed else ""

    def list_modified_files(self, repo_path: Path) -> List[str]:
        """List files currently modified or staged."""
        outcome = self.sandbox.execute_command("git diff --name-only", cwd=repo_path)
        if not outcome.passed or not outcome.stdout.strip():
            return []
        return [line.strip() for line in outcome.stdout.strip().splitlines() if line.strip()]

    def get_current_head_commit(self, repo_path: Path) -> Optional[str]:
        """Return the SHA-1 hash of the current HEAD commit."""
        outcome = self.sandbox.execute_command("git rev-parse HEAD", cwd=repo_path)
        if outcome.passed:
            return outcome.stdout.strip()
        return None
