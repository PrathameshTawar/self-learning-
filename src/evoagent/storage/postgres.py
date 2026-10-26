"""
PostgreSQL connection pool shared by the pgvector memory store and the job store.
"""

import atexit
from contextlib import contextmanager
import zlib
from typing import Any, Iterator


class PostgresDatabase:
    """
    A small psycopg 3 connection pool. Each `connection()` block is one transaction: it commits on
    success and rolls back on an exception.
    """

    def __init__(self, url: str, min_size: int = 1, max_size: int = 5) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError('PostgreSQL support needs: pip install -e ".[server]"') from exc
        if not url:
            raise ValueError("EVOAGENT_DATABASE_URL is not set")
        # psycopg takes libpq URLs; SQLAlchemy-style "postgresql+psycopg://" ones are accepted too
        self.url = url.replace("postgresql+psycopg://", "postgresql://", 1)
        self.pool = ConnectionPool(
            self.url, min_size=min_size, max_size=max_size, kwargs={"row_factory": dict_row}, open=True
        )
        # Short CLI commands exit without closing; an open pool would warn while its threads stop
        atexit.register(self.close)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self.pool.connection() as conn:
            yield conn

    def run_schema(self, name: str, statements: list[str]) -> None:
        """
        Apply idempotent DDL. Workers starting at the same moment would race on CREATE EXTENSION and
        CREATE TABLE, so the statements run under a transaction-scoped advisory lock.
        """
        with self.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (zlib.crc32(name.encode("utf-8")),))
            for statement in statements:
                conn.execute(statement)

    def ping(self) -> bool:
        try:
            with self.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if not self.pool.closed:
            self.pool.close()
