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
    provider = config.metadata_store.provider
    if provider == "sqlite":
        from memu.database.sqlite import build_sqlite_database

        return build_sqlite_database(config=config, user_model=user_model)
    msg = f"Unsupported metadata_store provider: {provider}"
    raise ValueError(msg)
