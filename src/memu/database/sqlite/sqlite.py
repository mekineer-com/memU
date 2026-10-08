"""SQLite database store implementation for MemU."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from memu.database.interfaces import Database
from memu.database.repositories import (
    CategoryItemRepo,
    DossierCandidateRepo,
    EntityRepo,
    MemoryCategoryRepo,
    MemoryItemRepo,
    ResourceRepo,
    TripleRepo,
)
from memu.database.sqlite.repositories.category_item_repo import SQLiteCategoryItemRepo
from memu.database.sqlite.repositories.dossier_candidate_repo import SQLiteDossierCandidateRepo
from memu.database.sqlite.repositories.entity_repo import SQLiteEntityRepo
from memu.database.sqlite.repositories.memory_category_repo import SQLiteMemoryCategoryRepo
from memu.database.sqlite.repositories.memory_item_repo import SQLiteMemoryItemRepo
from memu.database.sqlite.repositories.resource_repo import SQLiteResourceRepo
from memu.database.sqlite.repositories.triple_repo import SQLiteTripleRepo
from memu.database.sqlite.schema import SQLiteSQLAModels, get_sqlite_sqlalchemy_models
from memu.database.sqlite.session import SQLiteSessionManager

logger = logging.getLogger(__name__)


class SQLiteStore(Database):
    """SQLite database store implementation.

    This store provides a lightweight, file-based database backend for MemU.
    It uses SQLite for metadata storage and sqlite-vec cosine search.

    Attributes:
        resource_repo: Repository for resource records.
        memory_category_repo: Repository for memory categories.
        memory_item_repo: Repository for memory items.
        category_item_repo: Repository for category-item relations.
    """

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    dossier_candidate_repo: DossierCandidateRepo
    entity_repo: EntityRepo
    triple_repo: TripleRepo

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
        embedding_profile: str | None = None,
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
        self._sessions = SQLiteSessionManager(dsn=self.dsn)
        self._sessions.embedding_profile = embedding_profile
        self._sqla_models: SQLiteSQLAModels = sqla_models or get_sqlite_sqlalchemy_models(scope_model=self._scope_model)

        # Create tables
        self._create_tables()

        self._assert_canonical_embeddings()
        self._assert_embedding_profile()

        # Use provided models or defaults from sqla_models
        resource_model = resource_model or self._sqla_models.Resource
        memory_category_model = memory_category_model or self._sqla_models.MemoryCategory
        memory_item_model = memory_item_model or self._sqla_models.MemoryItem
        category_item_model = category_item_model or self._sqla_models.CategoryItem
        entity_model = self._sqla_models.Entity
        triple_model = self._sqla_models.Triple
        dossier_candidate_model = self._sqla_models.DossierCandidate

        # Initialize repositories
        self.resource_repo = SQLiteResourceRepo(
            resource_model=resource_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.memory_category_repo = SQLiteMemoryCategoryRepo(
            memory_category_model=memory_category_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.memory_item_repo = SQLiteMemoryItemRepo(
            memory_item_model=memory_item_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.category_item_repo = SQLiteCategoryItemRepo(
            category_item_model=category_item_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.dossier_candidate_repo = SQLiteDossierCandidateRepo(
            dossier_candidate_model=dossier_candidate_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.entity_repo = SQLiteEntityRepo(
            entity_model=entity_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )
        self.triple_repo = SQLiteTripleRepo(
            triple_model=triple_model,
            sqla_models=self._sqla_models,
            sessions=self._sessions,
            scope_fields=self._scope_fields,
        )

    def _assert_canonical_embeddings(self) -> None:
        invalid: list[str] = []
        with self._sessions.engine.connect() as conn:
            for table in ("resources", "memory_items", "categories"):
                text_count, empty_count, malformed_count = conn.exec_driver_sql(
                    f"SELECT "
                    "SUM(CASE WHEN embedding IS NOT NULL AND typeof(embedding) != 'blob' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN typeof(embedding) = 'blob' AND length(embedding) = 0 THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN typeof(embedding) = 'blob' AND length(embedding) % 4 != 0 THEN 1 ELSE 0 END) "
                    f"FROM {table}"
                ).one()
                if text_count or empty_count or malformed_count:
                    invalid.append(
                        f"{table}: non_blob={text_count or 0}, empty_blob={empty_count or 0}, "
                        f"malformed_blob={malformed_count or 0}"
                    )
        if invalid:
            detail = "; ".join(invalid)
            msg = f"non-canonical SQLite embeddings ({detail})"
            raise RuntimeError(msg)

    def _assert_embedding_profile(self) -> None:
        profile = self._sessions.embedding_profile
        if profile is None:
            return
        expected_bytes = int(profile.rsplit(":", 1)[1]) * 4
        with self._sessions.engine.connect() as conn:
            stored = conn.exec_driver_sql(
                "SELECT profile FROM embedding_profile WHERE id = 1"
            ).scalar()
            counts = {
                table: conn.exec_driver_sql(
                    f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL"
                ).scalar_one()
                for table in ("resources", "memory_items", "categories")
            }
            wrong_dimensions = {
                table: conn.exec_driver_sql(
                    f"SELECT COUNT(*) FROM {table} "
                    "WHERE embedding IS NOT NULL AND length(embedding) != ?",
                    (expected_bytes,),
                ).scalar_one()
                for table in counts
            }
        if stored is None and any(counts.values()):
            raise RuntimeError(
                f"populated database is missing embedding profile; expected {profile}"
            )
        if stored is not None and stored != profile:
            raise RuntimeError(
                f"embedding profile mismatch: database={stored} configured={profile}"
            )
        invalid = {table: count for table, count in wrong_dimensions.items() if count}
        if invalid:
            raise RuntimeError(f"embedding dimension mismatch for {profile}: {invalid}")

    def _ensure_fts_table(self) -> None:
        """Create FTS5 virtual table for BM25 keyword search on memory items."""
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

    def _create_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        self._sqla_models.Base.metadata.create_all(self._sessions.engine)
        self._ensure_fts_table()
        self._ensure_model_score_calibration_table()
        self._ensure_memory_item_edit_history_table()
        with self._sessions.engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS embedding_profile ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), profile TEXT NOT NULL)"
            )
        logger.debug("SQLite tables created/verified")

    def close(self) -> None:
        """Close the database connection and release resources."""
        self._sessions.close()


__all__ = ["SQLiteStore"]
