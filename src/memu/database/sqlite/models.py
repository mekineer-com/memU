"""SQLite-specific models for MemU database storage."""

from __future__ import annotations

import copy as _copy
import logging
import uuid
from datetime import datetime
from typing import Any

import pendulum
from pydantic import BaseModel
from sqlalchemy import JSON, Float, MetaData, String, Text
from sqlmodel import Column, DateTime, Field, Index, SQLModel, func

from memu.database.models import CategoryItem, Entity, MemoryCategory, MemoryItem, MemoryType, Resource, Triple

logger = logging.getLogger(__name__)


class TZDateTime(DateTime):
    """DateTime type with timezone support."""

    def __init__(self, timezone: bool = True, **kw: Any) -> None:
        super().__init__(timezone=timezone, **kw)


class SQLiteBaseModelMixin(SQLModel):
    """Base mixin for SQLite models with common fields."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True,
        sa_type=String,
    )
    created_at: datetime = Field(
        default_factory=lambda: pendulum.now("UTC"),
        sa_type=TZDateTime,
        sa_column_kwargs={"server_default": func.now()},
    )
    updated_at: datetime = Field(
        default_factory=lambda: pendulum.now("UTC"),
        sa_type=TZDateTime,
    )


class SQLiteResourceModel(SQLiteBaseModelMixin, Resource):
    """SQLite resource model."""

    url: str = Field(sa_column=Column(String, nullable=False))
    modality: str = Field(sa_column=Column(String, nullable=False))
    local_path: str = Field(sa_column=Column(String, nullable=False))
    caption: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Store embedding as JSON (SQLite stores it as TEXT under the hood)
    embedding: str | None = Field(default=None, sa_column=Column(Text, nullable=True))  # type: ignore[assignment]


class SQLiteMemoryItemModel(SQLiteBaseModelMixin, MemoryItem):
    """SQLite memory item model."""

    resource_id: str | None = Field(sa_column=Column(String, nullable=True))
    memory_type: MemoryType = Field(sa_column=Column(String, nullable=False))
    summary: str = Field(sa_column=Column(Text, nullable=False))
    # Store embedding as JSON (SQLite stores it as TEXT under the hood)
    embedding: str | None = Field(default=None, sa_column=Column(Text, nullable=True))  # type: ignore[assignment]
    happened_at: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    source_role: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    speaker_id: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    speaker_label: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    confidence: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    source_message_ids: list[int] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    reflection_salience: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    conversation_id: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    episode_id: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    unresolved: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    merged_into: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    extra: dict[str, Any] = Field(default={}, sa_column=Column(JSON, nullable=True))


class SQLiteMemoryCategoryModel(SQLiteBaseModelMixin, MemoryCategory):
    """SQLite memory category model."""

    name: str = Field(sa_column=Column(String, nullable=False, index=True))
    description: str = Field(sa_column=Column(Text, nullable=False))
    # Store embedding as JSON (SQLite stores it as TEXT under the hood)
    embedding: str | None = Field(default=None, sa_column=Column(Text, nullable=True))  # type: ignore[assignment]
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))


class SQLiteCategoryItemModel(SQLiteBaseModelMixin, CategoryItem):
    """SQLite category-item relation model."""

    item_id: str = Field(sa_column=Column(String, nullable=False))
    category_id: str = Field(sa_column=Column(String, nullable=False))

    # NOTE: SQLite reserves the "sqlite_" prefix for internal schema objects.
    __table_args__ = (Index("idx_memu_category_items_unique", "item_id", "category_id", unique=True),)


class SQLiteEntityModel(SQLiteBaseModelMixin, Entity):
    """SQLite entity model."""

    name: str = Field(sa_column=Column(String, nullable=False))
    entity_type: str = Field(sa_column=Column(String, nullable=False))
    normalized: str = Field(sa_column=Column(String, nullable=False, index=True))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON, nullable=True))


class SQLiteTripleModel(SQLiteBaseModelMixin, Triple):
    """SQLite triple model."""

    subject_id: str = Field(sa_column=Column(String, nullable=False))
    subject_kind: str = Field(sa_column=Column(String, nullable=False))
    predicate: str = Field(sa_column=Column(String, nullable=False))
    object_id: str = Field(sa_column=Column(String, nullable=False))
    object_kind: str = Field(sa_column=Column(String, nullable=False))
    valid_from: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    valid_to: datetime | None = Field(default=None, sa_column=Column(DateTime, nullable=True))
    confidence: float = Field(default=1.0, sa_column=Column(Float, nullable=True))
    source_memory_id: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON, nullable=True))

    __table_args__ = (
        Index("idx_memu_triples_subject", "subject_id"),
        Index("idx_memu_triples_object", "object_id"),
        Index("idx_memu_triples_predicate", "predicate"),
        Index("idx_memu_triples_predicate_subject", "predicate", "subject_id"),
        Index("idx_memu_triples_valid", "valid_from", "valid_to"),
    )


def _normalize_table_args(table_args: Any) -> tuple[list[Any], dict[str, Any]]:
    """Normalize SQLAlchemy table args to a consistent format."""
    if table_args is None:
        return [], {}
    if isinstance(table_args, dict):
        return [], dict(table_args)
    if not isinstance(table_args, tuple):
        return [table_args], {}

    args = list(table_args)
    kwargs: dict[str, Any] = {}
    if args and isinstance(args[-1], dict):
        kwargs = dict(args.pop())
    return args, kwargs


def _merge_models(
    user_model: type[BaseModel],
    core_model: type[SQLModel],
    *,
    name_suffix: str,
    base_attrs: dict[str, Any],
) -> type[SQLModel]:
    """Merge user scope model with core SQLModel."""
    overlap = set(user_model.model_fields) & set(core_model.model_fields)
    if overlap:
        msg = f"Scope fields conflict with core model fields: {sorted(overlap)}"
        raise TypeError(msg)

    return type(
        f"{user_model.__name__}{core_model.__name__}{name_suffix}",
        (user_model, core_model),
        base_attrs,
    )


def build_sqlite_table_model(
    user_model: type[BaseModel],
    core_model: type[SQLModel],
    *,
    tablename: str,
    metadata: MetaData | None = None,
    extra_table_args: tuple[Any, ...] | None = None,
    unique_with_scope: list[str] | None = None,
) -> type[SQLModel]:
    """Build a scoped SQLite table model."""
    overlap = set(user_model.model_fields) & set(core_model.model_fields)
    if overlap:
        msg = f"Scope fields conflict with core model fields: {sorted(overlap)}"
        raise TypeError(msg)

    scope_fields = list(user_model.model_fields.keys())
    base_table_args, table_kwargs = _normalize_table_args(getattr(core_model, "__table_args__", None))
    # Clone Index / UniqueConstraint / etc. — same single-Table binding rule as Columns.
    table_args = [_copy.deepcopy(arg) for arg in base_table_args]
    if extra_table_args:
        table_args.extend(extra_table_args)
    if scope_fields:
        table_args.append(Index(f"ix_{tablename}__scope", *scope_fields))
    if unique_with_scope:
        unique_cols = [*unique_with_scope, *scope_fields]
        table_args.append(Index(f"ix_{tablename}__unique_scoped", *unique_cols, unique=True))

    base_attrs: dict[str, Any] = {"__module__": core_model.__module__, "__tablename__": tablename}
    if metadata is not None:
        base_attrs["metadata"] = metadata
    if table_args or table_kwargs:
        if table_kwargs:
            base_attrs["__table_args__"] = (*table_args, table_kwargs)
        else:
            base_attrs["__table_args__"] = tuple(table_args)

    base = _merge_models(user_model, core_model, name_suffix="SQLiteBase", base_attrs=base_attrs)
    # Each scoped derivation gets fresh Column objects. SQLAlchemy binds a Column
    # to exactly one Table; without this, a second scope reusing the same Column
    # instance fails with "Column object already assigned to Table". Production
    # only uses one scope so this is test-hygiene in practice, but the original
    # code was quietly relying on that.
    _clone_scoped_sa_columns(base)

    # Use type() instead of create_model to properly preserve SQLModel table behavior
    table_attrs: dict[str, Any] = {"__module__": core_model.__module__}
    return type(
        f"{user_model.__name__}{core_model.__name__}SQLiteTable",
        (base,),
        table_attrs,
        table=True,
    )


def _clone_scoped_sa_columns(cls: type[SQLModel]) -> None:
    for fi in cls.model_fields.values():
        col = getattr(fi, "sa_column", None)
        if col is None:
            continue
        new_col = _copy.deepcopy(col)
        fi.sa_column = new_col
        # FieldInfoMetadata instances inside fi.metadata are shared with the core
        # model's FieldInfo. Replace the shared entry with a shallow copy carrying
        # the new Column — mutating the shared entry in place would corrupt the
        # core model for subsequent scoped builds.
        for i, meta_entry in enumerate(fi.metadata):
            if getattr(meta_entry, "sa_column", None) is col:
                new_entry = _copy.copy(meta_entry)
                new_entry.sa_column = new_col
                fi.metadata[i] = new_entry


__all__ = [
    "SQLiteBaseModelMixin",
    "SQLiteCategoryItemModel",
    "SQLiteEntityModel",
    "SQLiteMemoryCategoryModel",
    "SQLiteMemoryItemModel",
    "SQLiteResourceModel",
    "SQLiteTripleModel",
    "build_sqlite_table_model",
]
