"""SQLite database backend for MemU."""

from __future__ import annotations

from pydantic import BaseModel

from memu.app.settings import DatabaseConfig
from memu.database.sqlite.sqlite import SQLiteStore


def build_sqlite_database(
    *,
    config: DatabaseConfig,
    user_model: type[BaseModel],
) -> SQLiteStore:
    """Build a SQLite database store instance.

    Args:
        config: Database configuration containing metadata_store settings.
        user_model: Pydantic model for user scope fields.

    Returns:
        Configured SQLiteStore instance.
    """
    dsn = config.metadata_store.dsn
    if not dsn:
        # Default to a local file if no DSN provided
        dsn = "sqlite:///memu.db"

    return SQLiteStore(
        dsn=dsn,
        scope_model=user_model,
        embedding_profile=config.metadata_store.embedding_profile,
    )


__all__ = ["SQLiteStore", "build_sqlite_database"]
