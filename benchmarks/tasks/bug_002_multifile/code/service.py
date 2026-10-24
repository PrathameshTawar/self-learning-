from validator import normalize_and_validate


def process_order(order_id: str, description: str) -> dict:
    """Process order record by validating and formatting description."""
    cleaned = normalize_and_validate(description)
    return {
        "order_id": order_id.upper(),
        "description": cleaned,
        "is_valid": len(cleaned) > 0,
    }
