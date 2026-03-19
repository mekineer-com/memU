from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from memu.app.settings import DatabaseConfig
from memu.database.interfaces import Database

if TYPE_CHECKING:
    pass


def build_database(
    *,
    config: DatabaseConfig,
    user_model: type[BaseModel],
) -> Database:
    """
    Initialize a database backend for the configured provider.

    Supported providers:
        - "postgres": PostgreSQL with optional pgvector support
        - "sqlite": SQLite file-based storage (lightweight, portable)
    """
    provider = config.metadata_store.provider
    if provider == "postgres":
        # Lazy import to avoid requiring pgvector when not using postgres
        from memu.database.postgres import build_postgres_database

        return build_postgres_database(config=config, user_model=user_model)
    elif provider == "sqlite":
        # Lazy import to avoid loading SQLite dependencies when not needed
        from memu.database.sqlite import build_sqlite_database

        return build_sqlite_database(config=config, user_model=user_model)
    else:
        msg = f"Unsupported metadata_store provider: {provider}"
        raise ValueError(msg)
