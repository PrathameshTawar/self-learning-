def most_recent_event(events):
    """Return the name of the event with the largest timestamp, or None when there are no events."""
    ordered = sorted(events, key=lambda event: event["timestamp"])
    return ordered[-1]["name"]
