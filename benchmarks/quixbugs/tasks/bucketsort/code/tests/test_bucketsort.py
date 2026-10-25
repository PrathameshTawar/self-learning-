import pytest
from load_testdata import load_json_testcases

from bucketsort import bucketsort


testdata = load_json_testcases(bucketsort.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_bucketsort(input_data, expected):
    assert bucketsort(*input_data) == expected
