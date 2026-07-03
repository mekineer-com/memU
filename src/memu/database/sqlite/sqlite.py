"""SQLite database store implementation for MemU."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from sqlmodel import SQLModel

from memu.database.interfaces import Database
from memu.database.models import MemoryCategory, Resource
from memu.database.repositories import CategoryItemRepo, EntityRepo, MemoryCategoryRepo, MemoryItemRepo, ResourceRepo, TripleRepo
from memu.database.sqlite.repositories.category_item_repo import SQLiteCategoryItemRepo
from memu.database.sqlite.repositories.entity_repo import SQLiteEntityRepo
from memu.database.sqlite.repositories.memory_category_repo import SQLiteMemoryCategoryRepo
from memu.database.sqlite.repositories.memory_item_repo import SQLiteMemoryItemRepo
from memu.database.sqlite.repositories.resource_repo import SQLiteResourceRepo
from memu.database.sqlite.repositories.triple_repo import SQLiteTripleRepo
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
        categories: Dict cache of memory category records.
    """

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    entity_repo: EntityRepo
    triple_repo: TripleRepo
    resources: dict[str, Resource]
    categories: dict[str, MemoryCategory]

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
        entity_model = self._sqla_models.Entity
        triple_model = self._sqla_models.Triple

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
        self.entity_repo = SQLiteEntityRepo(
            state=self._state,
            entity_model=entity_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.triple_repo = SQLiteTripleRepo(
            state=self._state,
            triple_model=triple_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )

        # Set up cache references
        self.resources = self._state.resources
        self.categories = self._state.categories

    def _ensure_fts_table(self) -> None:
        """Create FTS5 virtual table for BM25 keyword search on memory items."""
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(
                    """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
USING fts5(
    summary,
    memory_type UNINDEXED,
    item_id UNINDEXED,
    tokenize='porter unicode61'
)
"""
                )
                fts_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_items_fts").scalar()
                if fts_count == 0:
                    items_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memory_items").scalar()
                    if items_count and items_count > 0:
                        conn.exec_driver_sql(
                            """
INSERT INTO memory_items_fts(summary, memory_type, item_id)
SELECT m.summary, m.memory_type, m.id
FROM memory_items AS m
WHERE (m.merged_into IS NULL OR TRIM(m.merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM triples AS t
    WHERE t.subject_id = m.id
      AND t.subject_kind = 'memory'
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
"""
                        )
                        logger.info("FTS5: backfilled %d items", items_count)
        except Exception:
            logger.warning("FTS5 table creation/backfill failed", exc_info=True)

    def _ensure_model_score_calibration_table(self) -> None:
        with self._sessions.engine.begin() as conn:
            conn.exec_driver_sql(
                """
CREATE TABLE IF NOT EXISTS model_score_calibration (
    model TEXT NOT NULL,
    field TEXT NOT NULL,
    version INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, field, version)
)
"""
            )

    def _ensure_memory_item_edit_history_table(self) -> None:
        with self._sessions.engine.begin() as conn:
            conn.exec_driver_sql(
                """
CREATE TABLE IF NOT EXISTS memory_item_edit_history (
    id TEXT PRIMARY KEY,
    memory_item_id TEXT NOT NULL,
    summary_before TEXT NOT NULL,
    summary_after TEXT NOT NULL,
    embedding_before TEXT,
    edited_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    edited_by TEXT,
    scope_json TEXT NOT NULL
)
"""
            )
            conn.exec_driver_sql(
                """
CREATE INDEX IF NOT EXISTS idx_memory_item_edit_history_item
ON memory_item_edit_history(memory_item_id, edited_at)
"""
            )

    def _ensure_category_previous_summary_column(self) -> None:
        with self._sessions.engine.begin() as conn:
            columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(categories)").fetchall()}
            if "previous_summary" not in columns:
                conn.exec_driver_sql("ALTER TABLE categories ADD COLUMN previous_summary TEXT")

    def _ensure_approval_columns(self) -> None:
        with self._sessions.engine.begin() as conn:
            memory_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(memory_items)").fetchall()}
            if "approved_at" not in memory_columns:
                conn.exec_driver_sql("ALTER TABLE memory_items ADD COLUMN approved_at DATETIME")
                conn.exec_driver_sql("UPDATE memory_items SET approved_at = CURRENT_TIMESTAMP")

            category_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(categories)").fetchall()}
            if "approved_summary" not in category_columns:
                conn.exec_driver_sql("ALTER TABLE categories ADD COLUMN approved_summary TEXT")
                conn.exec_driver_sql("UPDATE categories SET approved_summary = summary WHERE summary IS NOT NULL")

    def _create_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        SQLModel.metadata.create_all(self._sessions.engine)
        self._sqla_models.Base.metadata.create_all(self._sessions.engine)
        self._ensure_category_previous_summary_column()
        self._ensure_approval_columns()
        self._ensure_fts_table()
        self._ensure_model_score_calibration_table()
        self._ensure_memory_item_edit_history_table()
        logger.debug("SQLite tables created/verified")

    def close(self) -> None:
        """Close the database connection and release resources."""
        self._sessions.close()


__all__ = ["SQLiteStore"]
