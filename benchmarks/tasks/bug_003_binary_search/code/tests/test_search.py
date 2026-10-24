import pytest
from search import binary_search


def test_search_target_greater_or_at_end():
    # Targeted reproduction test: searching for elements beyond or at end triggers out-of-bounds IndexError
    assert binary_search([1, 3, 5], 10) == -1
    assert binary_search([2, 4, 6], 6) == 2


def test_empty_array():
    assert binary_search([], 5) == -1


def test_found_first():
    assert binary_search([10, 20, 30], 10) == 0


def test_found_middle():
    assert binary_search([10, 20, 30], 20) == 1
