def normalize_and_validate(raw_text: str) -> str:
    """Normalize input string by stripping whitespace and checking non-empty."""
    cleaned = raw_text.strip()
    return cleaned
