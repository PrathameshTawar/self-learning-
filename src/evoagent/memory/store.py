"""
Memory store implementations: SQLite persistent store and In-Memory test store.
Uses BM25 ranking for robust, order-independent failure trace and issue retrieval.
"""

import json
from pathlib import Path
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Optional
from evoagent.memory.bm25 import BM25Ranker
from evoagent.memory.schemas import MemoryEpisodeRecord


class IMemoryStore(ABC):
    """Abstract storage interface for episodic experience records."""

    @abstractmethod
    def save_episode(self, episode: MemoryEpisodeRecord) -> None:
        """Persist a completed task episode."""
        pass

    @abstractmethod
    def get_all_episodes(self, limit: int = 100) -> List[MemoryEpisodeRecord]:
        """Retrieve recent episodes."""
        pass

    @abstractmethod
    def get_accepted_episodes(self) -> List[MemoryEpisodeRecord]:
        """Retrieve only verified, successful repairs for dataset curation."""
        pass

    @abstractmethod
    def find_by_signature(self, signature: str, limit: int = 5) -> List[MemoryEpisodeRecord]:
        """Find past episodes matching a failure signature or exception type using BM25 ranking."""
        pass


class SQLiteMemoryStore(IMemoryStore):
    """Local SQLite-backed episodic memory with BM25 ranking."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ranker = BM25Ranker()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create episodes table if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    issue_description TEXT,
                    failure_signature TEXT,
                    target_file TEXT,
                    root_cause_hypothesis TEXT,
                    applied_patch TEXT,
                    outcome TEXT NOT NULL,
                    targeted_tests_passed INTEGER,
                    regression_tests_passed INTEGER,
                    lessons_learned TEXT,
                    model_version TEXT,
                    strategy_id TEXT,
                    created_at TEXT,
                    metadata_json TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signature ON episodes(failure_signature)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON episodes(outcome)")
            conn.commit()

    def save_episode(self, episode: MemoryEpisodeRecord) -> None:
        """Persist an episode into SQLite."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, task_id, issue_description, failure_signature,
                    target_file, root_cause_hypothesis, applied_patch, outcome,
                    targeted_tests_passed, regression_tests_passed, lessons_learned,
                    model_version, strategy_id, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.task_id,
                    episode.issue_description,
                    episode.failure_signature,
                    episode.target_file,
                    episode.root_cause_hypothesis,
                    episode.applied_patch,
                    episode.outcome,
                    1 if episode.targeted_tests_passed else 0,
                    1 if episode.regression_tests_passed else 0,
                    episode.lessons_learned,
                    episode.model_version,
                    episode.strategy_id,
                    episode.created_at.isoformat(),
                    json.dumps(episode.metadata),
                ),
            )
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryEpisodeRecord:
        return MemoryEpisodeRecord(
            episode_id=row["episode_id"],
            task_id=row["task_id"],
            issue_description=row["issue_description"] or "",
            failure_signature=row["failure_signature"] or "",
            target_file=row["target_file"] or "",
            root_cause_hypothesis=row["root_cause_hypothesis"] or "",
            applied_patch=row["applied_patch"] or "",
            outcome=row["outcome"],
            targeted_tests_passed=bool(row["targeted_tests_passed"]),
            regression_tests_passed=bool(row["regression_tests_passed"]),
            lessons_learned=row["lessons_learned"] or "",
            model_version=row["model_version"] or "baseline-001",
            strategy_id=row["strategy_id"] or "default",
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    def get_all_episodes(self, limit: int = 100) -> List[MemoryEpisodeRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_accepted_episodes(self) -> List[MemoryEpisodeRecord]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM episodes WHERE outcome = 'accepted' ORDER BY created_at DESC")
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def find_by_signature(self, signature: str, limit: int = 5) -> List[MemoryEpisodeRecord]:
        """Find past episodes using BM25 relevance scoring over stored episodes."""
        candidates = self.get_all_episodes(limit=200)
        return self.ranker.rank(query=signature, episodes=candidates, limit=limit)


class InMemoryMemoryStore(IMemoryStore):
    """Ephemeral memory store for isolated unit tests with BM25 ranking."""

    def __init__(self) -> None:
        self.episodes: List[MemoryEpisodeRecord] = []
        self.ranker = BM25Ranker()

    def save_episode(self, episode: MemoryEpisodeRecord) -> None:
        self.episodes.append(episode)

    def get_all_episodes(self, limit: int = 100) -> List[MemoryEpisodeRecord]:
        return list(reversed(self.episodes))[:limit]

    def get_accepted_episodes(self) -> List[MemoryEpisodeRecord]:
        return [e for e in self.episodes if e.outcome == "accepted"]

    def find_by_signature(self, signature: str, limit: int = 5) -> List[MemoryEpisodeRecord]:
        """Rank in-memory episodes using BM25."""
        return self.ranker.rank(query=signature, episodes=self.episodes, limit=limit)
