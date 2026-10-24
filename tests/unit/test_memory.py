"""
Unit tests for episodic memory persistence, experience retrieval, and BM25 ranking.
"""

from evoagent.domain.models import MemoryEpisodeRecord
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.memory.store import InMemoryMemoryStore


def test_memory_store_save_and_retrieve():
    store = InMemoryMemoryStore()
    ep = MemoryEpisodeRecord(
        episode_id="ep-1",
        task_id="bug-101",
        issue_description="Empty input IndexError in parser",
        failure_signature="IndexError: list index out of range",
        target_file="parser.py",
        root_cause_hypothesis="Empty list indexing",
        applied_patch="--- a/parser.py\n+++ b/parser.py",
        outcome="accepted",
        targeted_tests_passed=True,
        regression_tests_passed=True,
        lessons_learned="Check collection length before index 0 access",
    )
    store.save_episode(ep)

    retrieved = store.get_all_episodes()
    assert len(retrieved) == 1
    assert retrieved[0].task_id == "bug-101"

    accepted = store.get_accepted_episodes()
    assert len(accepted) == 1

    matches = store.find_by_signature("IndexError")
    assert len(matches) == 1
    assert matches[0].episode_id == "ep-1"


def test_experience_retriever_formats_lessons():
    store = InMemoryMemoryStore()
    store.save_episode(MemoryEpisodeRecord(
        episode_id="ep-2",
        task_id="bug-102",
        issue_description="Zero division in stats",
        failure_signature="ZeroDivisionError: division by zero",
        target_file="stats.py",
        root_cause_hypothesis="Denominator can be zero",
        applied_patch="return a / b if b != 0 else 0",
        outcome="accepted",
        targeted_tests_passed=True,
        regression_tests_passed=True,
        lessons_learned="Guard denominator against zero",
    ))

    retriever = ExperienceRetriever(store)
    lessons = retriever.retrieve_relevant_lessons("ZeroDivisionError")
    assert len(lessons) == 1
    assert "Guard denominator against zero" in lessons[0]


def test_bm25_memory_retrieval_ranking_and_order_independence():
    store = InMemoryMemoryStore()

    # Episode 1: Unrelated KeyError
    store.save_episode(MemoryEpisodeRecord(
        episode_id="ep-key",
        task_id="bug-201",
        issue_description="Missing config key in settings",
        failure_signature="KeyError: 'database'",
        target_file="config.py",
        root_cause_hypothesis="Key lookup without get default",
        applied_patch="return cfg.get('database', {})",
        outcome="accepted",
        targeted_tests_passed=True,
        regression_tests_passed=True,
        lessons_learned="Use .get() with defaults for optional keys",
    ))

    # Episode 2: Exact IndexError
    store.save_episode(MemoryEpisodeRecord(
        episode_id="ep-idx",
        task_id="bug-202",
        issue_description="Out of bounds list slicing on empty tokens",
        failure_signature="IndexError: list index out of range",
        target_file="tokenizer.py",
        root_cause_hypothesis="Empty tokens list accessed directly",
        applied_patch="return tokens[0] if tokens else ''",
        outcome="accepted",
        targeted_tests_passed=True,
        regression_tests_passed=True,
        lessons_learned="Guard empty tokens before index access",
    ))

    # Out-of-order query should still rank the IndexError episode #1
    matches = store.find_by_signature("out range index list empty tokens", limit=2)
    assert len(matches) >= 1
    assert matches[0].episode_id == "ep-idx"


def _episode(episode_id: str, signature: str, patch: str, outcome: str = "accepted") -> MemoryEpisodeRecord:
    return MemoryEpisodeRecord(
        episode_id=episode_id,
        task_id=f"task-{episode_id}",
        issue_description="",
        failure_signature=signature,
        target_file="mod.py",
        root_cause_hypothesis="",
        applied_patch=patch,
        outcome=outcome,
        targeted_tests_passed=outcome == "accepted",
        regression_tests_passed=outcome == "accepted",
        lessons_learned="lesson",
    )


def test_bm25_returns_nothing_for_unrelated_query():
    store = InMemoryMemoryStore()
    store.save_episode(_episode("ep-key", "KeyError: 'database'", "return cfg.get('database', {})"))

    assert store.find_by_signature("ZeroDivisionError division by zero") == []
    assert ExperienceRetriever(store).retrieve_relevant_lessons("ZeroDivisionError: division by zero") == []


def test_lesson_shows_changed_lines_of_search_replace_patch():
    patch = (
        "<<<<<<< SEARCH\n"
        "def ratio(a, b):\n"
        "    return a / b\n"
        "=======\n"
        "def ratio(a, b):\n"
        "    return a / b if b else 0.0\n"
        ">>>>>>> REPLACE"
    )
    store = InMemoryMemoryStore()
    store.save_episode(_episode("ep-div", "ZeroDivisionError: division by zero", patch))

    lessons = ExperienceRetriever(store).retrieve_relevant_lessons("ZeroDivisionError: division by zero")

    assert len(lessons) == 1
    assert "(Fix applied: return a / b if b else 0.0)" in lessons[0]
    assert "REPLACE" not in lessons[0]
