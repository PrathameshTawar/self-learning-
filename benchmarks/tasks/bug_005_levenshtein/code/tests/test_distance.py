import pytest
from distance import levenshtein_distance


def test_empty_string_boundary():
    # Targeted reproduction: when first string is empty, distance must equal len(s2)
    assert levenshtein_distance("", "kitten") == 6
    assert levenshtein_distance("", "a") == 1


def test_identical_strings():
    assert levenshtein_distance("algorithm", "algorithm") == 0


def test_single_edit():
    assert levenshtein_distance("cat", "hat") == 1
    assert levenshtein_distance("cat", "cats") == 1
