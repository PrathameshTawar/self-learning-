from typing import List


def is_rate_limited(
    history: List[float],
    current_time: float,
    window_seconds: float,
    limit: int,
) -> bool:
    """
    Determine whether a new request exceeds the sliding window rate limit.

    :param history: List of timestamps (epoch seconds) for earlier requests.
    :param current_time: Epoch timestamp of the current incoming request.
    :param window_seconds: Duration of the sliding window in seconds.
    :param limit: Maximum allowed requests within the sliding window.
    :return: True if rate limit is exceeded, False if request is allowed.
    """
    cutoff = current_time - window_seconds

    # Defect: filters timestamps older than cutoff instead of within active window
    valid_window_requests = [t for t in history if t < cutoff]

    return len(valid_window_requests) >= limit
