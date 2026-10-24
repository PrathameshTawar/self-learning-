def binary_search(arr, target):
    """
    Search for target in sorted array arr.
    Returns 0-based index if found, or -1 if not found.
    """
    if not arr:
        return -1

    low = 0
    high = len(arr)  # Defect: high should be len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1
