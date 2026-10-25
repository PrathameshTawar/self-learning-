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


def test_labeled_fenced_blocks_become_applicable_search_replace(tmp_path):
    from evoagent.domain.models import PatchProposal
    from evoagent.tools.patch_tools import PatchTool

    # Real Qwen2.5-3B output: headings over fenced code, indentation dropped
    raw = (
        "1. Hypothesis: the first row is initialized to 0.\n\n"
        "2. SEARCH:\n```python\nfor j in range(n + 1):\n    dp[0][j] = 0\n```\n\n"
        "3. REPLACE:\n```python\nfor j in range(n + 1):\n    dp[0][j] = j\n```"
    )
    patch = OpenAICompatibleModel()._extract_patch(raw)
    assert patch.startswith("<<<<<<< SEARCH\n")

    (tmp_path / "distance.py").write_text(
        "def build(n):\n    dp = [[0] * (n + 1)]\n    for j in range(n + 1):\n        dp[0][j] = 0\n    return dp\n",
        encoding="utf-8",
    )
    success, msg = PatchTool(root_path=tmp_path).apply_unified_diff(
        PatchProposal(target_file="distance.py", diff_unified=patch)
    )
    assert success, msg
    assert "        dp[0][j] = j\n" in (tmp_path / "distance.py").read_text(encoding="utf-8")


def test_request_caps_output_tokens(monkeypatch):
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

    OpenAICompatibleModel(max_tokens=256).generate_repair_proposal(_prompt())
    assert captured["max_tokens"] == 256


def test_no_api_key_sends_no_authorization_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": TWO_BLOCK_RESPONSE}}]})

    real_client = httpx.Client
    monkeypatch.setattr(
        open_model.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    # The default EVOAGENT_API_KEY is empty, as for a local Ollama server
    response = OpenAICompatibleModel(api_key="").generate_repair_proposal(_prompt())

    assert "authorization" not in captured["headers"]
    assert response.proposed_diff.count("<<<<<<< SEARCH") == 2
