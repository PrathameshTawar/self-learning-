def max_subarray_sum(nums):
    """Return the largest sum of any non-empty contiguous run of values in nums."""
    best = 0
    current = 0
    for value in nums:
        current = max(value, current + value)
        best = max(best, current)
    return best
