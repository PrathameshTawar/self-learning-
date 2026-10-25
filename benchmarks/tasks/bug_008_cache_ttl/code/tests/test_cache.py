from cache import fresh_entries


def test_returns_only_recent_entries():
    entries = [("a", 95.0), ("b", 10.0)]
    assert fresh_entries(entries, now=100.0, ttl_seconds=30.0) == [("a", 95.0)]


def test_empty_cache():
    assert fresh_entries([], now=100.0, ttl_seconds=30.0) == []
