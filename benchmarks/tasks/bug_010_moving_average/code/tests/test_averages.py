from averages import moving_average


def test_includes_last_window():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_empty_input():
    assert moving_average([], 3) == []


def test_fewer_values_than_window():
    assert moving_average([1], 3) == []
