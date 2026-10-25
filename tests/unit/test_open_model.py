"""
Unit tests for the OpenAI-compatible backend: strategy genes in the request and patch extraction.
"""

import json
import httpx
from evoagent.models import open_model
from evoagent.models.base import ModelPrompt
from evoagent.models.open_model import OpenAICompatibleModel

TWO_BLOCK_RESPONSE = (
    "Both helpers index an empty list.\n"
    "```python\n"
    "<<<<<<< SEARCH\n"
    "    return xs[0]\n"
    "=======\n"
    "    return xs[0] if xs else None\n"
    ">>>>>>> REPLACE\n"
    "<<<<<<< SEARCH\n"
    "    return xs[-1]\n"
    "=======\n"
    "    return xs[-1] if xs else None\n"
    ">>>>>>> REPLACE\n"
    "```\n"
)


def _prompt(**overrides) -> ModelPrompt:
    fields = dict(
        task_id="t1",
        issue_description="crash on empty list",
        failure_trace="IndexError: list index out of range",
        target_file="mod.py",
        file_content="def low(xs):\n    return xs[0]\n",
    )
    fields.update(overrides)
    return ModelPrompt(**fields)


def test_extract_patch_keeps_every_search_replace_block():
    patch = OpenAICompatibleModel()._extract_patch(TWO_BLOCK_RESPONSE)
    assert patch.count("<<<<<<< SEARCH") == 2
    assert "xs[-1] if xs else None" in patch


def test_request_carries_prompt_style_and_temperature(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": TWO_BLOCK_RESPONSE}}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        open_model.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    model = OpenAICompatibleModel(temperature=0.2)
    response = model.generate_repair_proposal(
        _prompt(system_instruction="Reasoning Mode: Chain-of-Thought.", temperature=0.6)
    )

    assert captured["temperature"] == 0.6
    assert "Reasoning Mode: Chain-of-Thought." in captured["messages"][0]["content"]
    assert response.proposed_diff.count("<<<<<<< SEARCH") == 2
    assert response.hypothesis == "Both helpers index an empty list."


def test_model_default_temperature_used_without_strategy(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "no patch"}}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        open_model.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    OpenAICompatibleModel(temperature=0.3).generate_repair_proposal(_prompt())
    assert captured["temperature"] == 0.3
