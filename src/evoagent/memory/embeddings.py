"""
Text embeddings for vector retrieval over episodic memory (pgvector backend).
"""

from abc import ABC, abstractmethod
import hashlib
import math
from typing import List
import httpx
from evoagent.memory.bm25 import tokenize
from evoagent.memory.schemas import MemoryEpisodeRecord


def episode_text(ep: MemoryEpisodeRecord) -> str:
    """Fields an episode is retrieved by, weighted like the BM25 document (signature counts most)."""
    return " ".join(
        [ep.failure_signature] * 3 + [ep.issue_description] * 2 + [ep.target_file] * 2 + [ep.root_cause_hypothesis]
    )


class IEmbedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an L2-normalized vector of length `dim`."""


class HashingEmbedder(IEmbedder):
    """
    Feature-hashed bag of unigrams and bigrams with sublinear term frequency. It needs no model or
    GPU and is deterministic, so every worker produces identical vectors; it captures lexical overlap
    only, which is what retrieval over error signatures relies on.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        # A second hash bit picks the sign, so colliding features cancel out instead of piling up
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> List[float]:
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for feature in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]:
            counts[feature] = counts.get(feature, 0) + 1
        vector = [0.0] * self.dim
        for feature, count in counts.items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class OpenAICompatibleEmbedder(IEmbedder):
    """Embeddings from an OpenAI-compatible /embeddings endpoint, e.g. Ollama's nomic-embed-text (768 dims)."""

    def __init__(self, base_url: str, model_name: str, dim: int, api_key: str = "", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.dim = dim
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        res = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model_name, "input": text},
            headers=headers,
            timeout=self.timeout,
        )
        res.raise_for_status()
        vector = res.json()["data"][0]["embedding"]
        if len(vector) != self.dim:
            raise ValueError(
                f"{self.model_name} returned {len(vector)}-dimensional embeddings; set EVOAGENT_EMBEDDING_DIM={len(vector)}"
            )
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


def create_embedder(provider: str, dim: int, model_name: str = "", base_url: str = "", api_key: str = "") -> IEmbedder:
    if provider == "openai_compatible":
        return OpenAICompatibleEmbedder(base_url, model_name, dim, api_key)
    return HashingEmbedder(dim)
