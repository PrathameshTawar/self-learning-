"""
Experience retrieval engine to provide historical lessons to the planner.
Supports configurable retrieval policies and top_k parameters from strategy chromosomes.
"""

import re
from typing import List, Optional
from evoagent.memory.bm25 import BM25Ranker
from evoagent.memory.schemas import MemoryEpisodeRecord
from evoagent.memory.store import IMemoryStore
from evoagent.tools.patch_tools import SEARCH_REPLACE_BLOCK_RE


def summarize_fix(patch: str, max_lines: int = 3) -> str:
    """Return the first few lines a patch introduces, for a compact lesson."""
    if SEARCH_REPLACE_BLOCK_RE.search(patch):
        new_lines = []
        for match in SEARCH_REPLACE_BLOCK_RE.finditer(patch):
            search_lines = set(match.group(1).splitlines())
            new_lines.extend(line for line in match.group(2).splitlines() if line not in search_lines)
    else:
        new_lines = [line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]

    new_lines = [line.strip() for line in new_lines if line.strip()]
    if not new_lines:
        # Plain-text patch without diff markers: keep its last line
        new_lines = [line.strip() for line in patch.splitlines() if line.strip()][-1:]
    return " | ".join(new_lines[:max_lines])


class ExperienceRetriever:
    """
    Retrieves and filters relevant historical repair lessons.
    Enables Layer A episodic learning by injecting past solutions into planner prompts.
    """

    def __init__(self, memory_store: IMemoryStore) -> None:
        self.memory_store = memory_store
        self.ranker = BM25Ranker()

    def retrieve_relevant_lessons(
        self,
        failure_signature: str,
        issue_description: str = "",
        failure_trace: str = "",
        max_results: int = 3,
        policy: str = "signature_match",
    ) -> List[str]:
        """
        Query memory store using the specified retrieval policy:
        - signature_match: Combines primary failure signature and issue description.
        - error_type_only: Isolates the Exception/Error class name for broad pattern retrieval.
        - bm25_semantic: Uses full failure trace stack and issue description for deep context.
        """
        policy_str = str(policy).lower()

        if "error_type" in policy_str:
            # Extract Exception name alone (e.g., IndexError, ZeroDivisionError)
            match = re.search(r"([A-Za-z]+Error|Exception)", failure_signature)
            query = match.group(1) if match else failure_signature
        elif "bm25" in policy_str or "semantic" in policy_str:
            # Deep failure trace search
            trace_snippet = "\n".join(failure_trace.strip().splitlines()[-6:]) if failure_trace else ""
            query = f"{failure_signature} {issue_description} {trace_snippet}".strip()
        else:
            # Standard signature match
            query = f"{failure_signature} {issue_description}".strip()

        # Only verified repairs become lessons. A failed episode records no fix, only that it failed;
        # in real runs those "caution" entries reached every prompt without adding information.
        episodes = self.ranker.rank(query, self.memory_store.get_accepted_episodes(), limit=max_results)
        return [
            f"[Task {ep.task_id} (Resolved)]: {ep.lessons_learned} (Fix applied: {summarize_fix(ep.applied_patch)})"
            for ep in episodes
            if ep.lessons_learned
        ]
