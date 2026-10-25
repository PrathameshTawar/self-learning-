from events import most_recent_event


def test_no_events_returns_none():
    assert most_recent_event([]) is None


def test_picks_latest_event():
    events = [{"name": "deploy", "timestamp": 20}, {"name": "build", "timestamp": 10}]
    assert most_recent_event(events) == "deploy"
