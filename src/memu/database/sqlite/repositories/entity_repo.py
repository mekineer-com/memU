"""SQLite entity repository implementation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from sqlmodel import select

from memu.database.models import Entity
from memu.database.repositories.entity import EntityRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState


def _normalize_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace to underscores."""
    return re.sub(r"\s+", "_", name.strip().lower())


class SQLiteEntityRepo(SQLiteRepoBase, EntityRepo):
    """SQLite implementation of entity repository."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        entity_model: type[Any],
        sqla_models: SQLiteSQLAModels,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        super().__init__(
            state=state,
            sqla_models=sqla_models,
            sessions=sessions,
            scope_fields=scope_fields,
        )
        self._entity_model = entity_model

    def _row_to_entity(self, row: Any) -> Entity:
        return Entity(
            id=row.id,
            name=row.name,
            entity_type=row.entity_type,
            normalized=row.normalized,
            properties=row.properties or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def get_or_create(
        self,
        name: str,
        entity_type: str,
        user_data: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Entity:
        normalized = _normalize_name(name)
        where = dict(user_data or {})
        create_scope = {k: v for k, v in where.items() if k in self._scope_fields and v is not None}
        if session is None:
            with self._sessions.session() as db_session:
                entity = self.get_or_create(name, entity_type, user_data=user_data, session=db_session)
                db_session.commit()
                return entity

        stmt = select(self._entity_model).where(self._entity_model.normalized == normalized)
        filters = self._build_filters(self._entity_model, where)
        if filters:
            stmt = stmt.where(*filters)
        row = session.exec(stmt).first()
        if row is not None:
            return self._row_to_entity(row)

        now = self._now()
        row = self._entity_model(
            name=name,
            entity_type=entity_type,
            normalized=normalized,
            properties={},
            created_at=now,
            updated_at=now,
            **create_scope,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        return self._row_to_entity(row)

    def lookup(self, normalized: str) -> Entity | None:
        with self._sessions.session() as session:
            stmt = select(self._entity_model).where(
                self._entity_model.normalized == normalized
            )
            row = session.exec(stmt).first()
            if row is None:
                return None
            return self._row_to_entity(row)

    def lookup_many(self, normalized_names: list[str]) -> list[Entity]:
        if not normalized_names:
            return []
        with self._sessions.session() as session:
            stmt = select(self._entity_model).where(
                self._entity_model.normalized.in_(normalized_names)
            )
            rows = session.exec(stmt).all()
            return [self._row_to_entity(r) for r in rows]

    def list_by_type(self, entity_type: str) -> list[Entity]:
        with self._sessions.session() as session:
            stmt = select(self._entity_model).where(
                self._entity_model.entity_type == entity_type
            )
            rows = session.exec(stmt).all()
            return [self._row_to_entity(r) for r in rows]

    def list_all(self, where: Mapping[str, Any] | None = None) -> list[Entity]:
        with self._sessions.session() as session:
            stmt = select(self._entity_model)
            filters = self._build_filters(self._entity_model, where)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()
            return [self._row_to_entity(r) for r in rows]


__all__ = ["SQLiteEntityRepo"]
