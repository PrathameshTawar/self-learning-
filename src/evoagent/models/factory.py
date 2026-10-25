"""
Model backend factory shared by the CLI, benchmark runner, and dashboard.
"""

from evoagent.models.base import IModelBackend
from evoagent.models.heuristic_model import HeuristicCodingModel
from evoagent.models.open_model import OpenAICompatibleModel

LLM_PROVIDERS = ("openai_compatible", "ollama")


def create_model_backend(
    provider: str, model_name: str, api_base_url: str, api_key: str, timeout: float = 120.0
) -> IModelBackend:
    """Return an LLM backend for OpenAI-compatible providers, otherwise the offline heuristic mock."""
    if provider in LLM_PROVIDERS:
        return OpenAICompatibleModel(base_url=api_base_url, model_name=model_name, api_key=api_key, timeout=timeout)
    return HeuristicCodingModel()
