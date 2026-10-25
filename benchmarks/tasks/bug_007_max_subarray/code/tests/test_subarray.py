from subarray import max_subarray_sum


def test_all_negative_values():
    assert max_subarray_sum([-3, -1, -2]) == -1


def test_mixed_values():
    assert max_subarray_sum([1, -2, 3, 4, -1]) == 7


def test_single_value():
    assert max_subarray_sum([5]) == 5
