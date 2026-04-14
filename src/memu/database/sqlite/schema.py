"""SQLAlchemy schema definitions for SQLite backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy import MetaData
from sqlmodel import SQLModel

from memu.database.sqlite.models import (
    SQLiteCategoryItemModel,
    SQLiteEntityModel,
    SQLiteMemoryCategoryModel,
    SQLiteMemoryItemModel,
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

    # NOTE: SQLite reserves the "sqlite_" prefix for internal schema objects.
    # Creating any table/index/view/trigger with that prefix fails with:
    #   sqlite3.OperationalError: object name reserved for internal use
    # Use a MemU-specific prefix instead.
    resource_model = build_sqlite_table_model(
        scope,
        SQLiteResourceModel,
        tablename="memu_resources",
        metadata=metadata_obj,
    )
    memory_category_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryCategoryModel,
        tablename="memu_memory_categories",
        metadata=metadata_obj,
    )
    memory_item_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryItemModel,
        tablename="memu_memory_items",
        metadata=metadata_obj,
    )
    category_item_model = build_sqlite_table_model(
        scope,
        SQLiteCategoryItemModel,
        tablename="memu_category_items",
        metadata=metadata_obj,
    )
    entity_model = build_sqlite_table_model(
        scope,
        SQLiteEntityModel,
        tablename="memu_entities",
        metadata=metadata_obj,
    )
    triple_model = build_sqlite_table_model(
        scope,
        SQLiteTripleModel,
        tablename="memu_triples",
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
        Entity=entity_model,
        Triple=triple_model,
    )
    _MODEL_CACHE[cache_key] = models
    return models


def get_sqlite_metadata(scope_model: type[BaseModel] | None = None) -> MetaData:
    """Get SQLAlchemy metadata for SQLite tables.

    Args:
        scope_model: Optional Pydantic model defining user scope fields.

    Returns:
        SQLAlchemy MetaData object.
    """
    from typing import cast

    return cast(MetaData, get_sqlite_sqlalchemy_models(scope_model=scope_model).Base.metadata)


__all__ = ["SQLiteSQLAModels", "get_sqlite_metadata", "get_sqlite_sqlalchemy_models"]
