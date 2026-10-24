import sys
from pathlib import Path

# Ensure code root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser import first_field


def test_first_field():
    """Verify standard extraction from comma-separated input."""
    assert first_field(" alpha,beta") == "alpha"


def test_empty_input():
    """Verify handling of empty or blank string."""
    assert first_field("") == ""
