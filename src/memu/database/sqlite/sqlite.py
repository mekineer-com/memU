"""SQLite database store implementation for MemU."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from sqlmodel import SQLModel

from memu.database.interfaces import Database
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, Resource
from memu.database.repositories import CategoryItemRepo, MemoryCategoryRepo, MemoryItemRepo, ResourceRepo
from memu.database.sqlite.repositories.category_item_repo import SQLiteCategoryItemRepo
from memu.database.sqlite.repositories.memory_category_repo import SQLiteMemoryCategoryRepo
from memu.database.sqlite.repositories.memory_item_repo import SQLiteMemoryItemRepo
from memu.database.sqlite.repositories.resource_repo import SQLiteResourceRepo
from memu.database.sqlite.schema import SQLiteSQLAModels, get_sqlite_sqlalchemy_models
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState

logger = logging.getLogger(__name__)


class SQLiteStore(Database):
    """SQLite database store implementation.

    This store provides a lightweight, file-based database backend for MemU.
    It uses SQLite for metadata storage and brute-force cosine similarity
    for vector search (native vector support is not available in SQLite).

    Attributes:
        resource_repo: Repository for resource records.
        memory_category_repo: Repository for memory categories.
        memory_item_repo: Repository for memory items.
        category_item_repo: Repository for category-item relations.
        resources: Dict cache of resource records.
        items: Dict cache of memory item records.
        categories: Dict cache of memory category records.
        relations: List cache of category-item relations.
    """

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    resources: dict[str, Resource]
    items: dict[str, MemoryItem]
    categories: dict[str, MemoryCategory]
    relations: list[CategoryItem]

    def __init__(
        self,
        *,
        dsn: str,
        scope_model: type[BaseModel] | None = None,
        resource_model: type[Any] | None = None,
        memory_category_model: type[Any] | None = None,
        memory_item_model: type[Any] | None = None,
        category_item_model: type[Any] | None = None,
        sqla_models: SQLiteSQLAModels | None = None,
    ) -> None:
        """Initialize SQLite database store.

        Args:
            dsn: SQLite connection string (e.g., "sqlite:///path/to/db.sqlite").
            scope_model: Pydantic model defining user scope fields.
            resource_model: Optional custom resource model.
            memory_category_model: Optional custom memory category model.
            memory_item_model: Optional custom memory item model.
            category_item_model: Optional custom category-item model.
            sqla_models: Pre-built SQLAlchemy models container.
        """
        self.dsn = dsn
        self._scope_model: type[BaseModel] = scope_model or BaseModel
        self._scope_fields = list(getattr(self._scope_model, "model_fields", {}).keys())
        self._state = DatabaseState()
        self._sessions = SQLiteSessionManager(dsn=self.dsn)
        self._sqla_models: SQLiteSQLAModels = sqla_models or get_sqlite_sqlalchemy_models(scope_model=self._scope_model)

        # Create tables
        self._create_tables()

        # Use provided models or defaults from sqla_models
        resource_model = resource_model or self._sqla_models.Resource
        memory_category_model = memory_category_model or self._sqla_models.MemoryCategory
        memory_item_model = memory_item_model or self._sqla_models.MemoryItem
        category_item_model = category_item_model or self._sqla_models.CategoryItem

        # Initialize repositories
        self.resource_repo = SQLiteResourceRepo(
            state=self._state,
            resource_model=resource_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.memory_category_repo = SQLiteMemoryCategoryRepo(
            state=self._state,
            memory_category_model=memory_category_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.memory_item_repo = SQLiteMemoryItemRepo(
            state=self._state,
            memory_item_model=memory_item_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.category_item_repo = SQLiteCategoryItemRepo(
            state=self._state,
            category_item_model=category_item_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )

        # Set up cache references
        self.resources = self._state.resources
        self.items = self._state.items
        self.categories = self._state.categories
        self.relations = self._state.relations

    def _ensure_embedding_json_columns(self) -> None:
        """Best-effort schema patch for legacy SQLite embedding compatibility.

        We no longer mirror writes into `embedding_json`, but we keep the
        legacy column available so older DB layouts continue to load cleanly.
        """
        tables = ["memu_resources", "memu_memory_items", "memu_memory_categories"]
        try:
            with self._sessions.engine.begin() as conn:
                for tname in tables:
                    rows = conn.exec_driver_sql(f"PRAGMA table_info({tname})").fetchall()
                    cols = [r[1] for r in rows] if rows else []
                    if not cols:
                        continue
                    if "embedding_json" not in cols:
                        conn.exec_driver_sql(f"ALTER TABLE {tname} ADD COLUMN embedding_json TEXT")
        except Exception:
            # Best-effort only; don't block startup.
            return

    @staticmethod
    def _table_columns(conn: Any, table_name: str) -> list[str]:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        return [r[1] for r in rows] if rows else []

    def _add_column_if_missing(self, conn: Any, table_name: str, column_name: str, ddl: str) -> bool:
        cols = self._table_columns(conn, table_name)
        if not cols or column_name in cols:
            return False
        conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")
        return True

    def _ensure_memory_item_provenance_columns(self) -> None:
        """Best-effort migration for Phase 1 provenance columns."""
        try:
            with self._sessions.engine.begin() as conn:
                self._add_column_if_missing(conn, "memu_memory_items", "source_role", "source_role VARCHAR")
                self._add_column_if_missing(conn, "memu_memory_items", "confidence", "confidence REAL")
                self._add_column_if_missing(conn, "memu_memory_items", "conversation_id", "conversation_id VARCHAR")
                self._add_column_if_missing(conn, "memu_memory_items", "merged_into", "merged_into VARCHAR")

                cols = self._table_columns(conn, "memu_memory_items")
                # Backfill conversation_id from legacy session_id when available.
                if "conversation_id" in cols and "session_id" in cols:
                    conn.exec_driver_sql(
                        "UPDATE memu_memory_items "
                        "SET conversation_id = session_id "
                        "WHERE (conversation_id IS NULL OR TRIM(conversation_id) = '') "
                        "AND session_id IS NOT NULL AND TRIM(session_id) <> ''"
                    )
        except Exception:
            return

    def _ensure_conversation_state_table(self) -> None:
        """Create/patch service-level conversation state table."""
        create_sql = """
CREATE TABLE IF NOT EXISTS memu_conversation_state (
    conversation_id VARCHAR PRIMARY KEY,
    agent_id VARCHAR,
    user_id VARCHAR,
    digest_cursor INTEGER DEFAULT 0,
    working_note TEXT,
    active_intentions JSON,
    last_retrieval_ids JSON,
    last_memorize_at DATETIME,
    updated_at DATETIME
)
"""
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(create_sql)
                self._add_column_if_missing(conn, "memu_conversation_state", "agent_id", "agent_id VARCHAR")
                self._add_column_if_missing(conn, "memu_conversation_state", "user_id", "user_id VARCHAR")
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "digest_cursor", "digest_cursor INTEGER DEFAULT 0"
                )
                self._add_column_if_missing(conn, "memu_conversation_state", "working_note", "working_note TEXT")
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "active_intentions", "active_intentions JSON"
                )
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "last_retrieval_ids", "last_retrieval_ids JSON"
                )
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "last_memorize_at", "last_memorize_at DATETIME"
                )
                self._add_column_if_missing(conn, "memu_conversation_state", "updated_at", "updated_at DATETIME")
        except Exception:
            return

    def _create_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        SQLModel.metadata.create_all(self._sessions.engine)
        # Also create tables from our custom metadata
        self._sqla_models.Base.metadata.create_all(self._sessions.engine)
        # Patch up mixed embedding columns on existing DBs.
        self._ensure_embedding_json_columns()
        self._ensure_memory_item_provenance_columns()
        self._ensure_conversation_state_table()
        logger.debug("SQLite tables created/verified")

    def close(self) -> None:
        """Close the database connection and release resources."""
        self._sessions.close()

    def load_existing(self) -> None:
        """Load all existing data from database into cache."""
        self.resource_repo.load_existing()
        self.memory_category_repo.load_existing()
        self.memory_item_repo.load_existing()
        self.category_item_repo.load_existing()


__all__ = ["SQLiteStore"]
