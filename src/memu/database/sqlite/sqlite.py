"""SQLite database store implementation for MemU."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from sqlmodel import SQLModel

from memu.database.interfaces import Database
from memu.database.models import CategoryItem, MemoryCategory, MemoryItem, Resource
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
        items: Dict cache of memory item records.
        categories: Dict cache of memory category records.
        relations: List cache of category-item relations.
    """

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    entity_repo: EntityRepo
    triple_repo: TripleRepo
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
        self.items = self._state.items
        self.categories = self._state.categories
        self.relations = self._state.relations

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

    def _ensure_conversation_state_table(self) -> None:
        """Create/patch service-level conversation state table."""
        create_sql = """
CREATE TABLE IF NOT EXISTS memu_conversation_state (
    conversation_id VARCHAR PRIMARY KEY,
    soul_id VARCHAR,
    user_id VARCHAR,
    digest_cursor INTEGER DEFAULT 0,
    prior_context TEXT,
    active_intentions JSON,
    memory_cache JSON DEFAULT '[]',
    pending_episode_ids JSON DEFAULT '[]',
    self_model_id VARCHAR,
    last_retrieval_ids JSON,
    last_memorize_at DATETIME,
    updated_at DATETIME
)
"""
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(create_sql)
                self._add_column_if_missing(conn, "memu_conversation_state", "soul_id", "soul_id VARCHAR")
                self._add_column_if_missing(conn, "memu_conversation_state", "user_id", "user_id VARCHAR")
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "digest_cursor", "digest_cursor INTEGER DEFAULT 0"
                )
                self._add_column_if_missing(conn, "memu_conversation_state", "prior_context", "prior_context TEXT")
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "active_intentions", "active_intentions JSON"
                )
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "memory_cache", "memory_cache JSON DEFAULT '[]'"
                )
                self._add_column_if_missing(
                    conn,
                    "memu_conversation_state",
                    "pending_episode_ids",
                    "pending_episode_ids JSON DEFAULT '[]'",
                )
                self._add_column_if_missing(conn, "memu_conversation_state", "self_model_id", "self_model_id VARCHAR")
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "last_retrieval_ids", "last_retrieval_ids JSON"
                )
                self._add_column_if_missing(
                    conn, "memu_conversation_state", "last_memorize_at", "last_memorize_at DATETIME"
                )
                self._add_column_if_missing(conn, "memu_conversation_state", "updated_at", "updated_at DATETIME")
                self._add_column_if_missing(
                    conn,
                    "memu_conversation_state",
                    "retrieve_rewrite_angle",
                    "retrieve_rewrite_angle INTEGER DEFAULT 0",
                )
        except Exception:
            return

    def _ensure_self_model_tables(self) -> None:
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(
                    """
CREATE TABLE IF NOT EXISTS memu_self_model (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    narrative_self TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
                )
                self._add_column_if_missing(conn, "memu_self_model", "related_memory_ids", "related_memory_ids TEXT")
                cols = set(self._table_columns(conn, "memu_self_model"))
                if "trait_invariants" in cols or "contextual_state" in cols:
                    conn.exec_driver_sql(
                        """
CREATE TABLE memu_self_model__new (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    narrative_self TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
                    )
                    conn.exec_driver_sql(
                        """
INSERT INTO memu_self_model__new (id, soul_id, user_id, narrative_self, related_memory_ids, updated_at)
SELECT id, soul_id, user_id, narrative_self, related_memory_ids, updated_at
FROM memu_self_model
"""
                    )
                    conn.exec_driver_sql("DROP TABLE memu_self_model")
                    conn.exec_driver_sql("ALTER TABLE memu_self_model__new RENAME TO memu_self_model")
                conn.exec_driver_sql(
                    """
CREATE TABLE IF NOT EXISTS memu_intentions (
    id TEXT PRIMARY KEY,
    soul_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT,
    confidence REAL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    target_date TEXT,
    related_memory_ids TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
                )
        except Exception:
            return

    def _ensure_fts_table(self) -> None:
        """Create FTS5 virtual table for BM25 keyword search on memory items."""
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(
                    """
CREATE VIRTUAL TABLE IF NOT EXISTS memu_memory_items_fts
USING fts5(
    summary,
    memory_type UNINDEXED,
    item_id UNINDEXED,
    tokenize='porter unicode61'
)
"""
                )
                # Backfill if FTS table is empty but items exist
                fts_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memu_memory_items_fts").scalar()
                if fts_count == 0:
                    items_count = conn.exec_driver_sql("SELECT COUNT(*) FROM memu_memory_items").scalar()
                    if items_count and items_count > 0:
                        conn.exec_driver_sql(
                            """
INSERT INTO memu_memory_items_fts(summary, memory_type, item_id)
SELECT m.summary, m.memory_type, m.id
FROM memu_memory_items AS m
WHERE (m.merged_into IS NULL OR TRIM(m.merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM memu_triples AS t
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

    def _ensure_triple_indexes(self) -> None:
        try:
            with self._sessions.engine.begin() as conn:
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_memu_triples_predicate_subject ON memu_triples(predicate, subject_id)"
                )
        except Exception:
            logger.warning("Triple index creation failed", exc_info=True)

    @staticmethod
    def _missing_scope_expr(column: str) -> str:
        return f"({column} IS NULL OR TRIM({column}) = '')"

    def _resolve_scope_fallback(self, conn: Any, fields: list[str]) -> dict[str, str]:
        if not fields:
            return {}
        required = " AND ".join([f"m.{f} IS NOT NULL AND TRIM(m.{f}) != ''" for f in fields])
        select_fields = ", ".join([f"m.{f}" for f in fields])
        sql = (
            f"SELECT {select_fields} FROM memu_memory_items m "
            f"WHERE {required} GROUP BY {select_fields} LIMIT 2"
        )
        rows = conn.exec_driver_sql(sql).fetchall()
        if len(rows) != 1:
            return {}
        row = rows[0]
        out: dict[str, str] = {}
        for idx, field in enumerate(fields):
            value = row[idx]
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out[field] = text
        return out

    @staticmethod
    def _ensure_migrations_table(conn: Any) -> None:
        conn.exec_driver_sql(
            """
CREATE TABLE IF NOT EXISTS memu_migrations (
    name TEXT PRIMARY KEY,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
        )

    def _migration_applied(self, conn: Any, name: str) -> bool:
        self._ensure_migrations_table(conn)
        row = conn.exec_driver_sql(
            "SELECT 1 FROM memu_migrations WHERE name = :name LIMIT 1",
            {"name": name},
        ).fetchone()
        return row is not None

    def _mark_migration_applied(self, conn: Any, name: str) -> None:
        self._ensure_migrations_table(conn)
        conn.exec_driver_sql(
            "INSERT OR IGNORE INTO memu_migrations(name) VALUES (:name)",
            {"name": name},
        )

    def _backfill_graph_scope(self) -> None:
        scope_fields = [f for f in self._scope_fields if f in {"user_id", "soul_id"}]
        if not scope_fields:
            return
        migration_name = "graph_scope_backfill_v1"
        try:
            with self._sessions.engine.begin() as conn:
                if self._migration_applied(conn, migration_name):
                    return
                table_names = {
                    str(row[0])
                    for row in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if row and row[0]
                }
                if "memu_memory_items" not in table_names:
                    self._mark_migration_applied(conn, migration_name)
                    return
                entity_cols = set(self._table_columns(conn, "memu_entities")) if "memu_entities" in table_names else set()
                triple_cols = set(self._table_columns(conn, "memu_triples")) if "memu_triples" in table_names else set()

                # First pass: infer triple scope from source memory rows when missing.
                if "memu_triples" in table_names and triple_cols:
                    for field in scope_fields:
                        if field not in triple_cols:
                            continue
                        missing = self._missing_scope_expr(f"t.{field}")
                        source_ok = f"m.{field} IS NOT NULL AND TRIM(m.{field}) != ''"
                        conn.exec_driver_sql(
                            f"""
UPDATE memu_triples AS t
SET {field} = (
  SELECT m.{field}
  FROM memu_memory_items AS m
  WHERE m.id = t.source_memory_id
    AND {source_ok}
  LIMIT 1
)
WHERE {missing}
  AND t.source_memory_id IS NOT NULL
"""
                        )

                # Second pass: infer entity scope from mentions triples when missing.
                if "memu_entities" in table_names and entity_cols and "memu_triples" in table_names and triple_cols:
                    for field in scope_fields:
                        if field not in entity_cols or field not in triple_cols:
                            continue
                        missing = self._missing_scope_expr(f"e.{field}")
                        source_ok = f"t.{field} IS NOT NULL AND TRIM(t.{field}) != ''"
                        conn.exec_driver_sql(
                            f"""
UPDATE memu_entities AS e
SET {field} = (
  SELECT t.{field}
  FROM memu_triples AS t
  WHERE t.object_id = e.id
    AND t.predicate = 'mentions'
    AND t.object_kind = 'entity'
    AND {source_ok}
  LIMIT 1
)
WHERE {missing}
"""
                        )

                # Final pass: any remaining missing scope gets the DB fallback scope.
                fallback = self._resolve_scope_fallback(conn, scope_fields)
                if fallback:
                    for table_name, cols in (("memu_entities", entity_cols), ("memu_triples", triple_cols)):
                        if table_name not in table_names or not cols:
                            continue
                        for field in scope_fields:
                            if field not in cols:
                                continue
                            value = fallback.get(field)
                            if not value:
                                continue
                            conn.exec_driver_sql(
                                f"""
UPDATE {table_name}
SET {field} = :value
WHERE {self._missing_scope_expr(field)}
""",
                                {"value": value},
                            )
                self._mark_migration_applied(conn, migration_name)
        except Exception:
            logger.warning("Graph scope backfill failed", exc_info=True)

    def _create_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        SQLModel.metadata.create_all(self._sessions.engine)
        # Also create tables from our custom metadata
        self._sqla_models.Base.metadata.create_all(self._sessions.engine)
        self._ensure_conversation_state_table()
        self._ensure_self_model_tables()
        self._ensure_triple_indexes()
        self._ensure_fts_table()
        self._backfill_graph_scope()
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
