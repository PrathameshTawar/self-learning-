import sys
from pathlib import Path

# Add code directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from service import process_order


def test_process_standard_order():
    """Verify processing with standard valid input."""
    result = process_order("ord-123", "  rush delivery  ")
    assert result["order_id"] == "ORD-123"
    assert result["description"] == "rush delivery"
    assert result["is_valid"] is True


def test_process_none_input():
    """Verify processing gracefully handles None description without AttributeError."""
    result = process_order("ord-456", None)
    assert result["order_id"] == "ORD-456"
    assert result["description"] == ""
    assert result["is_valid"] is False
