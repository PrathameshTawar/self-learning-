"""
Okapi BM25 ranking algorithm for episodic memory retrieval over error signatures and descriptions.
"""

import math
import re
from typing import List, Tuple
from evoagent.memory.schemas import MemoryEpisodeRecord

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
}


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens, filtering out stop words."""
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


class BM25Ranker:
    """
    Okapi BM25 implementation for scoring memory episodes against failure signatures,
    issue descriptions, and code symbols.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _episode_to_tokens(self, ep: MemoryEpisodeRecord) -> List[str]:
        # Weighted document construction
        sig_tokens = tokenize(ep.failure_signature) * 3
        desc_tokens = tokenize(ep.issue_description) * 2
        file_tokens = tokenize(ep.target_file) * 2
        hyp_tokens = tokenize(ep.root_cause_hypothesis)
        return sig_tokens + desc_tokens + file_tokens + hyp_tokens

    def rank(
        self,
        query: str,
        episodes: List[MemoryEpisodeRecord],
        limit: int = 5,
    ) -> List[MemoryEpisodeRecord]:
        """Rank episodes by BM25 relevance to the query."""
        if not episodes:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        corpus_tokens = [self._episode_to_tokens(ep) for ep in episodes]
        N = len(episodes)
        avgdl = sum(len(d) for d in corpus_tokens) / max(1, N)

        # Document frequency for query terms
        df = {}
        for q in query_tokens:
            df[q] = sum(1 for doc in corpus_tokens if q in doc)

        scored_episodes: List[Tuple[float, MemoryEpisodeRecord]] = []

        for idx, (doc, ep) in enumerate(zip(corpus_tokens, episodes)):
            doc_len = len(doc)
            if doc_len == 0:
                continue

            score = 0.0
            term_freq = {}
            for t in doc:
                term_freq[t] = term_freq.get(t, 0) + 1

            for q in query_tokens:
                n_q = df.get(q, 0)
                if n_q == 0 or q not in term_freq:
                    continue

                # Standard Okapi BM25 IDF formula
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                tf = term_freq[q]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                score += idf * (numerator / denominator)

            # Boost accepted resolution episodes over rejected episodes
            if ep.outcome == "accepted":
                score *= 1.25

            if score > 0.0:
                scored_episodes.append((score, ep))

        # Sort descending by BM25 score. Episodes sharing no terms with the query are left out:
        # injecting unrelated "lessons" only distracts the model.
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_episodes[:limit]]
