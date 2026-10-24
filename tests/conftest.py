"""
Pytest configuration and shared test fixtures for EvoAgent.
"""

import sys
from pathlib import Path
import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evoagent.memory.store import InMemoryMemoryStore
from evoagent.sandbox.local_runner import LocalSubprocessSandbox


@pytest.fixture
def in_memory_store() -> InMemoryMemoryStore:
    """Fixture providing an ephemeral in-memory episode store."""
    return InMemoryMemoryStore()


@pytest.fixture
def local_sandbox() -> LocalSubprocessSandbox:
    """Fixture providing a subprocess execution sandbox."""
    return LocalSubprocessSandbox()


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Fixture creating a temporary repository with a reproducible bug and tests."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create buggy parser.py
    code_file = repo / "parser.py"
    code_file.write_text(
        "def first_field(line: str) -> str:\n"
        "    parts = [p for p in line.split(',') if p]\n"
        "    return parts[0].strip()\n",
        encoding="utf-8",
    )

    # Create test suite
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_parser.py"
    test_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "from parser import first_field\n\n"
        "def test_first_field():\n"
        "    assert first_field(' alpha,beta') == 'alpha'\n\n"
        "def test_empty_input():\n"
        "    assert first_field('') == ''\n",
        encoding="utf-8",
    )

    return repo
