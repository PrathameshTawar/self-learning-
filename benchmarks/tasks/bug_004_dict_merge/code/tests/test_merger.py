import pytest
from merger import deep_merge_config


def test_deep_merge_with_none_override():
    # Targeted reproduction: override is None, which triggers AttributeError
    base = {"database": {"host": "localhost", "port": 5432}}
    merged = deep_merge_config(base, None)
    assert merged == base


def test_deep_merge_flat_dicts():
    base = {"a": 1, "b": 2}
    override = {"b": 20, "c": 30}
    assert deep_merge_config(base, override) == {"a": 1, "b": 20, "c": 30}


def test_deep_merge_nested_dicts():
    base = {"app": {"name": "app", "debug": False}}
    override = {"app": {"debug": True}}
    assert deep_merge_config(base, override) == {"app": {"name": "app", "debug": True}}
