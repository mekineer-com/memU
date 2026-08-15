"""SQLite entity repository implementation."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from sqlalchemy import text
from sqlmodel import select

from memu.database.models import Entity, normalize_entity_name
from memu.database.repositories.entity import EntityRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState


def _clean_aliases(values: list[str], primary_name: str) -> list[str]:
    seen = {normalize_entity_name(primary_name)}
    aliases: list[str] = []
    for value in values:
        alias = str(value or "").strip()
        normalized = normalize_entity_name(alias)
        if alias and normalized not in seen:
            seen.add(normalized)
            aliases.append(alias)
    return aliases


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
        self._entity_write_lock = threading.RLock()

    @contextmanager
    def write_lock(self):
        with self._entity_write_lock:
            yield

    @staticmethod
    def _begin_write(session: Any) -> None:
        if not session.in_transaction():
            session.execute(text("BEGIN IMMEDIATE"))

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

    def _get_or_create_in_session(
        self,
        *,
        name: str,
        entity_type: str,
        normalized: str,
        where: Mapping[str, Any],
        create_scope: Mapping[str, Any],
        session: Any,
    ) -> Entity:
        stmt = select(self._entity_model).where(self._entity_model.normalized == normalized)
        filters = self._build_filters(self._entity_model, where)
        if filters:
            stmt = stmt.where(*filters)
        row = session.exec(stmt).first()
        if row is not None:
            return self._row_to_entity(row)

        # ponytail: aliases live in JSON, so rare name misses scan one soul's entities.
        alias_matches = []
        alias_stmt = select(self._entity_model)
        if filters:
            alias_stmt = alias_stmt.where(*filters)
        for candidate in session.exec(alias_stmt).all():
            if candidate.entity_type != entity_type:
                continue
            aliases = (candidate.properties or {}).get("aliases", [])
            if isinstance(aliases, list) and any(
                normalize_entity_name(str(alias or "")) == normalized for alias in aliases
            ):
                alias_matches.append(candidate)
        if len(alias_matches) == 1:
            return self._row_to_entity(alias_matches[0])

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

    def get_or_create(
        self,
        name: str,
        entity_type: str,
        user_data: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Entity:
        normalized = normalize_entity_name(name)
        where = dict(user_data or {})
        create_scope = {k: v for k, v in where.items() if k in self._scope_fields and v is not None}
        if session is None:
            with self._sessions.session() as db_session:
                entity = self.get_or_create(name, entity_type, where, session=db_session)
                db_session.commit()
                return entity
        self._begin_write(session)
        with self._entity_write_lock:
            return self._get_or_create_in_session(
                name=name,
                entity_type=entity_type,
                normalized=normalized,
                where=where,
                create_scope=create_scope,
                session=session,
            )

    def create(
        self,
        name: str,
        entity_type: str,
        user_data: Mapping[str, Any],
        *,
        properties: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Entity:
        scope = self._require_scope(user_data)
        if session is None:
            with self._sessions.session() as db_session:
                entity = self.create(
                    name,
                    entity_type,
                    scope,
                    properties=properties,
                    session=db_session,
                )
                db_session.commit()
                return entity
        self._begin_write(session)
        now = self._now()
        clean_properties = dict(properties or {})
        clean_aliases = _clean_aliases(list(clean_properties.get("aliases") or []), name)
        if clean_aliases:
            clean_properties["aliases"] = clean_aliases
        else:
            clean_properties.pop("aliases", None)
        with self._entity_write_lock:
            row = self._entity_model(
                name=name,
                entity_type=entity_type,
                normalized=normalize_entity_name(name),
                properties=clean_properties,
                created_at=now,
                updated_at=now,
                **scope,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return self._row_to_entity(row)

    def update(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any],
        name: str | None = None,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
        property_updates: Mapping[str, Any] | None = None,
        property_removals: set[str] | None = None,
        session: Any | None = None,
    ) -> Entity:
        scope = self._require_scope(where)
        if session is None:
            with self._sessions.session() as db_session:
                entity = self.update(
                    entity_id,
                    where=scope,
                    name=name,
                    entity_type=entity_type,
                    aliases=aliases,
                    property_updates=property_updates,
                    property_removals=property_removals,
                    session=db_session,
                )
                db_session.commit()
                return entity
        self._begin_write(session)
        with self._entity_write_lock:
            stmt = select(self._entity_model).where(self._entity_model.id == entity_id)
            filters = self._build_filters(self._entity_model, scope)
            if filters:
                stmt = stmt.where(*filters)
            row = session.exec(stmt).first()
            if row is None:
                raise KeyError(f"entity not found in scope: {entity_id}")

            properties = dict(row.properties or {})
            next_name = name.strip() if name is not None else str(row.name)
            if not next_name:
                raise ValueError("entity name is required")
            next_normalized = normalize_entity_name(next_name)
            next_aliases = list(properties.get("aliases") or []) if aliases is None else list(aliases)
            if next_normalized != row.normalized:
                next_aliases.append(str(row.name))
            clean_aliases = _clean_aliases(next_aliases, next_name)
            if clean_aliases:
                properties["aliases"] = clean_aliases
            else:
                properties.pop("aliases", None)
            properties.update(property_updates or {})
            for key in property_removals or set():
                properties.pop(key, None)

            next_type = entity_type if entity_type is not None else row.entity_type
            if (
                next_name == row.name
                and next_normalized == row.normalized
                and next_type == row.entity_type
                and properties == (row.properties or {})
            ):
                return self._row_to_entity(row)
            row.name = next_name
            row.normalized = next_normalized
            row.entity_type = next_type
            row.properties = properties
            row.updated_at = self._now()
            session.add(row)
            session.flush()
            session.refresh(row)
            return self._row_to_entity(row)

    def list_all(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[Entity]:
        stmt = select(self._entity_model)
        filters = self._build_filters(self._entity_model, where)
        if filters:
            stmt = stmt.where(*filters)
        if session is not None:
            return [self._row_to_entity(row) for row in session.exec(stmt).all()]
        with self._sessions.session() as db_session:
            return [self._row_to_entity(row) for row in db_session.exec(stmt).all()]

    def delete(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any],
        session: Any | None = None,
    ) -> None:
        scope = self._require_scope(where)
        if session is None:
            with self._sessions.session() as db_session:
                self.delete(entity_id, where=scope, session=db_session)
                db_session.commit()
                return
        self._begin_write(session)
        with self._entity_write_lock:
            stmt = select(self._entity_model).where(self._entity_model.id == entity_id)
            filters = self._build_filters(self._entity_model, scope)
            if filters:
                stmt = stmt.where(*filters)
            row = session.exec(stmt).first()
            if row is None:
                raise KeyError(f"entity not found in scope: {entity_id}")
            session.delete(row)
            session.flush()

    def list_by_ids(
        self,
        entity_ids: set[str],
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[Entity]:
        if not entity_ids:
            return []
        stmt = select(self._entity_model).where(self._entity_model.id.in_(entity_ids))
        filters = self._build_filters(self._entity_model, where)
        if filters:
            stmt = stmt.where(*filters)
        if session is not None:
            return [self._row_to_entity(row) for row in session.exec(stmt).all()]
        with self._sessions.session() as db_session:
            return [self._row_to_entity(row) for row in db_session.exec(stmt).all()]

    def _bind_source_refs_in_session(
        self,
        bindings: Mapping[str, str],
        where: Mapping[str, Any],
        session: Any,
    ) -> list[Entity]:
        stmt = select(self._entity_model)
        filters = self._build_filters(self._entity_model, where)
        if filters:
            stmt = stmt.where(*filters)
        rows = session.exec(stmt).all()
        rows_by_id = {str(row.id): row for row in rows}
        owners: dict[str, str] = {}
        for row in rows:
            refs = (row.properties or {}).get("source_refs", [])
            if not isinstance(refs, list):
                raise ValueError(f"entity {row.id} source_refs must be a list")
            for ref in refs:
                normalized = str(ref or "").strip()
                owner = owners.get(normalized)
                if normalized and owner and owner != str(row.id):
                    raise ValueError(f"source reference {normalized!r} belongs to multiple entities")
                if normalized:
                    owners[normalized] = str(row.id)

        for source_ref, entity_id in bindings.items():
            owner = owners.get(source_ref)
            if owner and owner != entity_id:
                raise ValueError(f"source reference {source_ref!r} already belongs to entity {owner}")
            if entity_id not in rows_by_id:
                raise ValueError(f"source reference target entity not found in scope: {entity_id}")

        now = self._now()
        for source_ref, entity_id in bindings.items():
            row = rows_by_id[entity_id]
            properties = dict(row.properties or {})
            refs = list(properties.get("source_refs") or [])
            if source_ref not in refs:
                properties["source_refs"] = [*refs, source_ref]
                row.properties = properties
                row.updated_at = now
                session.add(row)
        session.flush()
        return [self._row_to_entity(row) for row in rows]

    def bind_source_refs(
        self,
        bindings: Mapping[str, str],
        where: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> list[Entity]:
        if session is None:
            with self._sessions.session() as db_session:
                entities = self.bind_source_refs(bindings, where, session=db_session)
                db_session.commit()
                return entities
        self._begin_write(session)
        with self._entity_write_lock:
            return self._bind_source_refs_in_session(bindings, dict(where or {}), session)


__all__ = ["SQLiteEntityRepo"]
