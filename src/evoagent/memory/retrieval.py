"""
Experience retrieval engine to provide historical lessons to the planner.
"""

from typing import List
from evoagent.memory.schemas import MemoryEpisodeRecord
from evoagent.memory.store import IMemoryStore


class ExperienceRetriever:
    """
    Retrieves and filters relevant historical repair lessons.
    Enables Layer A episodic learning by injecting past solutions into planner prompts.
    """

    def __init__(self, memory_store: IMemoryStore) -> None:
        self.memory_store = memory_store

    def retrieve_relevant_lessons(
        self,
        failure_signature: str,
        issue_description: str = "",
        max_results: int = 3,
    ) -> List[str]:
        """Query memory store and return formatted, verified lessons."""
        query = f"{failure_signature} {issue_description}".strip()
        episodes = self.memory_store.find_by_signature(query, limit=max_results * 2)

        lessons = []
        for ep in episodes:
            if ep.outcome == "accepted" and ep.lessons_learned:
                lessons.append(
                    f"[Task {ep.task_id} (Resolved)]: {ep.lessons_learned} (Fix applied: {ep.applied_patch.splitlines()[-1] if ep.applied_patch else ''})"
                )
            elif ep.outcome == "rejected" and ep.lessons_learned:
                lessons.append(
                    f"[Task {ep.task_id} (Caution)]: Past failed attempt warning - {ep.lessons_learned}"
                )

            if len(lessons) >= max_results:
                break

        return lessons
