def deep_merge_config(base, override):
    """
    Recursively merges override dictionary into base dictionary.
    Returns a newly merged dictionary.
    """
    if base is None:
        return override if isinstance(override, dict) else {}

    result = dict(base)

    # Defect: does not guard against override being None before calling .items()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value

    return result
