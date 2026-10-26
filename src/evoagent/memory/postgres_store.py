"""
PostgreSQL + pgvector episodic memory, shared by every worker so each one learns from all of them.
"""

import json
from typing import Any, Dict, List
from evoagent.memory.bm25 import BM25Ranker
from evoagent.memory.embeddings import IEmbedder, episode_text
from evoagent.memory.schemas import MemoryEpisodeRecord
from evoagent.memory.store import IMemoryStore
from evoagent.storage.postgres import PostgresDatabase

COLUMNS = (
    "episode_id, task_id, issue_description, failure_signature, target_file, root_cause_hypothesis, "
    "applied_patch, outcome, targeted_tests_passed, regression_tests_passed, lessons_learned, "
    "model_version, strategy_id, created_at, metadata"
)


def vector_literal(vector: List[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class PostgresMemoryStore(IMemoryStore):
    """
    Episodes live in one table with an embedding column. Retrieval first takes the nearest verified
    episodes from an HNSW index, then BM25 ranks that shortlist exactly as the SQLite store does, so
    switching backends changes how candidates are found, not how lessons are ranked.
    """

    def __init__(self, db: PostgresDatabase, embedder: IEmbedder) -> None:
        self.db = db
        self.embedder = embedder
        self.ranker = BM25Ranker()
        self._init_schema()

    def _init_schema(self) -> None:
        dim = int(self.embedder.dim)
        self.db.run_schema("memory_episodes", [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""
            CREATE TABLE IF NOT EXISTS memory_episodes (
                episode_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                issue_description TEXT NOT NULL DEFAULT '',
                failure_signature TEXT NOT NULL DEFAULT '',
                target_file TEXT NOT NULL DEFAULT '',
                root_cause_hypothesis TEXT NOT NULL DEFAULT '',
                applied_patch TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL,
                targeted_tests_passed BOOLEAN NOT NULL,
                regression_tests_passed BOOLEAN NOT NULL,
                lessons_learned TEXT NOT NULL DEFAULT '',
                model_version TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}',
                embedding vector({dim}) NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS memory_episodes_created_idx ON memory_episodes (created_at DESC)",
            # Partial index: retrieval only ever searches verified episodes, and filtering after an
            # HNSW scan could otherwise return fewer rows than asked for
            "CREATE INDEX IF NOT EXISTS memory_episodes_accepted_hnsw ON memory_episodes "
            "USING hnsw (embedding vector_cosine_ops) WHERE outcome = 'accepted'",
        ])
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'memory_episodes'::regclass AND attname = 'embedding'"
            ).fetchone()
        if row and row["atttypmod"] != dim:
            raise RuntimeError(
                f"memory_episodes stores {row['atttypmod']}-dimensional embeddings but the embedder makes {dim}. "
                "Set EVOAGENT_EMBEDDING_DIM to match, or migrate the table."
            )

    @staticmethod
    def _to_record(row: Dict[str, Any]) -> MemoryEpisodeRecord:
        return MemoryEpisodeRecord(**{**row, "metadata": row["metadata"] or {}})

    def save_episode(self, episode: MemoryEpisodeRecord) -> None:
        with self.db.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO memory_episodes ({COLUMNS}, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (episode_id) DO UPDATE SET
                    outcome = EXCLUDED.outcome, applied_patch = EXCLUDED.applied_patch,
                    lessons_learned = EXCLUDED.lessons_learned, metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    episode.episode_id, episode.task_id, episode.issue_description, episode.failure_signature,
                    episode.target_file, episode.root_cause_hypothesis, episode.applied_patch, episode.outcome,
                    episode.targeted_tests_passed, episode.regression_tests_passed, episode.lessons_learned,
                    episode.model_version, episode.strategy_id, episode.created_at, json.dumps(episode.metadata),
                    vector_literal(self.embedder.embed(episode_text(episode))),
                ),
            )

    def get_all_episodes(self, limit: int = 100) -> List[MemoryEpisodeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT {COLUMNS} FROM memory_episodes ORDER BY created_at DESC LIMIT %s", (limit,)
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def get_accepted_episodes(self) -> List[MemoryEpisodeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT {COLUMNS} FROM memory_episodes WHERE outcome = 'accepted' ORDER BY created_at DESC"
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def candidate_episodes(self, query: str, limit: int = 100) -> List[MemoryEpisodeRecord]:
        """Nearest verified episodes by cosine distance, as the shortlist BM25 then ranks."""
        with self.db.connection() as conn:
            # ef_search bounds how many rows an HNSW scan can return (default 40)
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, limit)),))
            rows = conn.execute(
                f"SELECT {COLUMNS} FROM memory_episodes WHERE outcome = 'accepted' "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector_literal(self.embedder.embed(query)), limit),
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def find_by_signature(self, signature: str, limit: int = 5) -> List[MemoryEpisodeRecord]:
        with self.db.connection() as conn:
            rows = conn.execute(
                f"SELECT {COLUMNS} FROM memory_episodes ORDER BY embedding <=> %s::vector LIMIT 200",
                (vector_literal(self.embedder.embed(signature)),),
            ).fetchall()
        return self.ranker.rank(query=signature, episodes=[self._to_record(r) for r in rows], limit=limit)

    def count_episodes(self) -> Dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT outcome, count(*) AS n FROM memory_episodes GROUP BY outcome").fetchall()
        return {r["outcome"]: r["n"] for r in rows}
