"""SQLAlchemy schema definitions for SQLite backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Index, MetaData
from sqlmodel import SQLModel

from memu.database.sqlite.models import (
    SQLiteCategoryItemModel,
    SQLiteDossierCandidateModel,
    SQLiteEntityModel,
    SQLiteMemoryCategoryModel,
    SQLiteMemoryItemModel,
    SQLiteMemoryRefCounterModel,
    SQLiteResourceModel,
    SQLiteTripleModel,
    build_sqlite_table_model,
)


@dataclass
class SQLiteSQLAModels:
    """Container for SQLite SQLAlchemy/SQLModel models."""

    Base: type[Any]
    Resource: type[Any]
    MemoryCategory: type[Any]
    MemoryItem: type[Any]
    CategoryItem: type[Any]
    DossierCandidate: type[Any]
    MemoryRefCounter: type[Any]
    Entity: type[Any]
    Triple: type[Any]


_MODEL_CACHE: dict[type[Any], SQLiteSQLAModels] = {}


def get_sqlite_sqlalchemy_models(*, scope_model: type[BaseModel] | None = None) -> SQLiteSQLAModels:
    """Build (and cache) SQLModel ORM models for SQLite storage.

    Args:
        scope_model: Optional Pydantic model defining user scope fields.

    Returns:
        SQLiteSQLAModels containing all table models.
    """
    scope = scope_model or BaseModel
    cache_key = scope
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    metadata_obj = MetaData()
    scope_fields = list(scope.model_fields)

    resource_model = build_sqlite_table_model(
        scope,
        SQLiteResourceModel,
        tablename="resources",
        metadata=metadata_obj,
    )
    memory_category_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryCategoryModel,
        tablename="categories",
        metadata=metadata_obj,
        extra_table_args=(
            Index("ix_categories__anchor_scoped", *scope_fields, "anchor_role", unique=True),
            Index("ix_categories__activity_scoped", *scope_fields, "kind", "last_evidence_at"),
            Index("ix_categories__entity_scoped", *scope_fields, "entity_id"),
        ),
    )
    memory_item_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryItemModel,
        tablename="memory_items",
        metadata=metadata_obj,
        extra_table_args=(
            Index("ix_memory_items__ref_scoped", *scope_fields, "memory_ref", unique=True),
        ),
    )
    category_item_model = build_sqlite_table_model(
        scope,
        SQLiteCategoryItemModel,
        tablename="category_items",
        metadata=metadata_obj,
        extra_table_args=(
            Index("ix_category_items__category_scoped", *scope_fields, "category_id"),
        ),
    )
    dossier_candidate_model = build_sqlite_table_model(
        scope,
        SQLiteDossierCandidateModel,
        tablename="dossier_candidates",
        metadata=metadata_obj,
        extra_table_args=(
            Index(
                "ix_dossier_candidates__unique_scoped",
                *scope_fields,
                "normalized_name",
                "item_id",
                unique=True,
            ),
        ),
    )
    memory_ref_counter_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryRefCounterModel,
        tablename="memory_ref_counters",
        metadata=metadata_obj,
        extra_table_args=(
            Index(
                "ix_memory_ref_counters__unique_scoped",
                *scope_fields,
                "counter_key",
                unique=True,
            ),
        ),
    )
    entity_model = build_sqlite_table_model(
        scope,
        SQLiteEntityModel,
        tablename="entities",
        metadata=metadata_obj,
    )
    triple_model = build_sqlite_table_model(
        scope,
        SQLiteTripleModel,
        tablename="triples",
        metadata=metadata_obj,
    )

    class SQLiteBase(SQLModel):
        __abstract__ = True
        metadata = metadata_obj

    models = SQLiteSQLAModels(
        Base=SQLiteBase,
        Resource=resource_model,
        MemoryCategory=memory_category_model,
        MemoryItem=memory_item_model,
        CategoryItem=category_item_model,
        DossierCandidate=dossier_candidate_model,
        MemoryRefCounter=memory_ref_counter_model,
        Entity=entity_model,
        Triple=triple_model,
    )
    _MODEL_CACHE[cache_key] = models
    return models


__all__ = ["SQLiteSQLAModels", "get_sqlite_sqlalchemy_models"]
