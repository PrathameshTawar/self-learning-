"""
Shared PostgreSQL connection handling for the production backends.
"""

from evoagent.storage.postgres import PostgresDatabase

__all__ = ["PostgresDatabase"]
