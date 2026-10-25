def moving_average(values, window):
    """Return the average of every complete run of `window` consecutive values."""
    return [sum(values[i:i + window]) / window for i in range(len(values) - window)]
