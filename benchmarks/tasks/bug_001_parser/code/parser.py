def first_field(line: str) -> str:
    """Extract and trim the first comma-separated field from an input line."""
    parts = [p for p in line.split(",") if p]
    return parts[0].strip()
