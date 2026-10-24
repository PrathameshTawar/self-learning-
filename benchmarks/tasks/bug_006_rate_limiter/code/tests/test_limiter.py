import pytest
from limiter import is_rate_limited


def test_requests_exceeding_rate_limit_rejected():
    # Targeted reproduction: 3 requests within 10s window with limit 2 must be rate limited
    history = [95.0, 98.0, 99.0]
    current_time = 100.0
    assert is_rate_limited(history, current_time, window_seconds=10.0, limit=2) is True


def test_empty_history_allowed():
    assert is_rate_limited([], current_time=100.0, window_seconds=10.0, limit=2) is False


def test_expired_history_allowed():
    # Requests at t=10.0 and t=20.0 are outside 10s window at t=100.0
    history = [10.0, 20.0]
    assert is_rate_limited(history, current_time=100.0, window_seconds=10.0, limit=2) is False
