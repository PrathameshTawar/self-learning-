def fresh_entries(entries, now, ttl_seconds):
    """Return the (key, created_at) cache entries that are younger than ttl_seconds at time now."""
    return [entry for entry in entries if now - entry[1] > ttl_seconds]
