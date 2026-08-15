"""SQLite triple repository implementation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import or_
from sqlmodel import select

from memu.database.models import Triple
from memu.database.repositories.triple import TripleRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState

# Predicates whose meaning is direction-independent: (A,p,B) == (B,p,A).
# For these, endpoints are stored in sorted order so two writes from
# opposite sides dedup into one row. Callers that need to see "all
# things p-related to X" must query both subject and object directions
# (get_connected_memory_edges already does this).
SYMMETRIC_PREDICATES = frozenset({"conflicts_with", "parallels"})


def _canonical_endpoints(predicate: str, subject_id: str, object_id: str) -> tuple[str, str]:
    if predicate in SYMMETRIC_PREDICATES and subject_id > object_id:
        return object_id, subject_id
    return subject_id, object_id


class SQLiteTripleRepo(SQLiteRepoBase, TripleRepo):
    """SQLite implementation of triple repository."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        triple_model: type[Any],
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
        self._triple_model = triple_model

    def _row_to_triple(self, row: Any) -> Triple:
        return Triple(
            id=row.id,
            subject_id=row.subject_id,
            subject_kind=row.subject_kind,
            predicate=row.predicate,
            object_id=row.object_id,
            object_kind=row.object_kind,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            confidence=row.confidence,
            source_memory_id=row.source_memory_id,
            properties=row.properties or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def add(
        self,
        triple: Triple,
        user_data: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Triple:
        now = self._now()
        create_scope = {k: v for k, v in dict(user_data or {}).items() if k in self._scope_fields and v is not None}
        if session is None:
            with self._sessions.session() as db_session:
                persisted = self.add(triple, user_data=user_data, session=db_session)
                db_session.commit()
                return persisted

        subject_id, object_id = _canonical_endpoints(
            triple.predicate, triple.subject_id, triple.object_id
        )

        # Skip dedup for historical triples (valid_to set on creation) —
        # they don't compete for the "current" row.
        if triple.valid_to is None:
            stmt = select(self._triple_model).where(
                self._triple_model.subject_id == subject_id,
                self._triple_model.predicate == triple.predicate,
                self._triple_model.object_id == object_id,
                self._triple_model.valid_to.is_(None),
            )
            scope_filters = self._build_filters(self._triple_model, create_scope)
            if scope_filters:
                stmt = stmt.where(*scope_filters)
            existing = session.exec(stmt).first()
            if existing is not None:
                return self._row_to_triple(existing)

        row = self._triple_model(
            id=triple.id,
            subject_id=subject_id,
            subject_kind=triple.subject_kind,
            predicate=triple.predicate,
            object_id=object_id,
            object_kind=triple.object_kind,
            valid_from=triple.valid_from or now,
            valid_to=triple.valid_to,
            confidence=triple.confidence,
            source_memory_id=triple.source_memory_id,
            properties=triple.properties,
            created_at=now,
            updated_at=now,
            **create_scope,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        return self._row_to_triple(row)

    def get_edges_from(
        self,
        subject_id: str,
        predicate: str | None = None,
        current_only: bool = True,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[Triple]:
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                self._triple_model.subject_id == subject_id
            )
            filters = self._build_filters(self._triple_model, where)
            if filters:
                stmt = stmt.where(*filters)
            if predicate is not None:
                stmt = stmt.where(self._triple_model.predicate == predicate)
            if as_of is not None:
                stmt = stmt.where(
                    self._triple_model.valid_from <= as_of,
                    (self._triple_model.valid_to.is_(None)) | (self._triple_model.valid_to >= as_of),
                )
            elif current_only:
                stmt = stmt.where(self._triple_model.valid_to.is_(None))
            rows = session.exec(stmt).all()
            return [self._row_to_triple(r) for r in rows]

    def get_edges_to(
        self,
        object_id: str,
        predicate: str | None = None,
        current_only: bool = True,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[Triple]:
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                self._triple_model.object_id == object_id
            )
            filters = self._build_filters(self._triple_model, where)
            if filters:
                stmt = stmt.where(*filters)
            if predicate is not None:
                stmt = stmt.where(self._triple_model.predicate == predicate)
            if as_of is not None:
                stmt = stmt.where(
                    self._triple_model.valid_from <= as_of,
                    (self._triple_model.valid_to.is_(None)) | (self._triple_model.valid_to >= as_of),
                )
            elif current_only:
                stmt = stmt.where(self._triple_model.valid_to.is_(None))
            rows = session.exec(stmt).all()
            return [self._row_to_triple(r) for r in rows]

    def list_edges_for_memories(
        self,
        memory_ids: Collection[str],
        predicates: Collection[str],
        where: Mapping[str, Any] | None = None,
        *,
        current_only: bool = True,
    ) -> list[Triple]:
        ids = set(memory_ids)
        predicate_set = set(predicates)
        if not ids or not predicate_set:
            return []
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                or_(
                    self._triple_model.subject_id.in_(ids),
                    self._triple_model.object_id.in_(ids),
                ),
                self._triple_model.predicate.in_(predicate_set),
            )
            filters = self._build_filters(self._triple_model, where)
            if filters:
                stmt = stmt.where(*filters)
            if current_only:
                stmt = stmt.where(self._triple_model.valid_to.is_(None))
            rows = session.exec(stmt).all()
        return [self._row_to_triple(row) for row in rows]

    def invalidate(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        scope: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> None:
        now = self._now()
        subject_id, object_id = _canonical_endpoints(predicate, subject_id, object_id)
        if session is None:
            with self._sessions.session() as db_session:
                self.invalidate(subject_id, predicate, object_id, scope, session=db_session)
                db_session.commit()
            return
        stmt = select(self._triple_model).where(
            self._triple_model.subject_id == subject_id,
            self._triple_model.predicate == predicate,
            self._triple_model.object_id == object_id,
            self._triple_model.valid_to.is_(None),
        )
        scope_filters = self._build_filters(self._triple_model, scope)
        if scope_filters:
            stmt = stmt.where(*scope_filters)
        for row in session.exec(stmt).all():
            row.valid_to = now
            row.updated_at = now
            session.add(row)
        session.flush()

    def replace_entity_id(
        self,
        discarded_id: str,
        canonical_id: str,
        where: Mapping[str, Any],
        session: Any,
    ) -> int:
        scope = self._require_scope(where)
        stmt = select(self._triple_model).where(
            or_(
                (self._triple_model.subject_kind == "entity")
                & (self._triple_model.subject_id == discarded_id),
                (self._triple_model.object_kind == "entity")
                & (self._triple_model.object_id == discarded_id),
            ),
            *self._build_filters(self._triple_model, scope),
        )
        rows = list(session.exec(stmt).all())
        now = self._now()
        for row in rows:
            subject_id = canonical_id if row.subject_kind == "entity" and row.subject_id == discarded_id else row.subject_id
            object_id = canonical_id if row.object_kind == "entity" and row.object_id == discarded_id else row.object_id
            row.subject_id, row.object_id = _canonical_endpoints(row.predicate, subject_id, object_id)
            row.updated_at = now
            session.add(row)

        session.flush()
        impacted_keys = {(row.subject_id, row.predicate, row.object_id) for row in rows}

        current_stmt = select(self._triple_model).where(
            self._triple_model.valid_to.is_(None),
            *self._build_filters(self._triple_model, scope),
        )
        current_rows = list(session.exec(current_stmt).all())
        groups: dict[tuple[str, str, str], list[Any]] = {}
        for row in current_rows:
            key = (row.subject_id, row.predicate, row.object_id)
            if key in impacted_keys:
                groups.setdefault(key, []).append(row)
        for group in groups.values():
            group.sort(key=lambda row: (str(row.valid_from or ""), str(row.created_at or ""), str(row.id)))
            for row in group[1:]:
                row.valid_to = now
                row.updated_at = now
                session.add(row)
        for row in rows:
            if (
                row.valid_to is None
                and row.subject_kind == "entity"
                and row.object_kind == "entity"
                and row.subject_id == row.object_id
            ):
                row.valid_to = now
                row.updated_at = now
                session.add(row)
        session.flush()
        return len(rows)

    def list_entity_references(
        self,
        entity_id: str,
        where: Mapping[str, Any],
        session: Any,
    ) -> list[Triple]:
        scope = self._require_scope(where)
        rows = session.exec(
            select(self._triple_model).where(
                or_(
                    (self._triple_model.subject_kind == "entity")
                    & (self._triple_model.subject_id == entity_id),
                    (self._triple_model.object_kind == "entity")
                    & (self._triple_model.object_id == entity_id),
                ),
                *self._build_filters(self._triple_model, scope),
            )
        ).all()
        return [self._row_to_triple(row) for row in rows]

    def get_connected_memory_edges(
        self,
        memory_ids: list[str],
        predicates: list[str] | None = None,
        max_per_source: int = 3,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[tuple[str, str, str]]:
        if not memory_ids:
            return []

        seen: set[str] = set()
        result: list[tuple[str, str, str]] = []

        with self._sessions.session() as session:
            filters = self._build_filters(self._triple_model, where)
            for mid in memory_ids:
                # Outgoing edges from this memory
                stmt_out = select(self._triple_model.object_id, self._triple_model.predicate).where(
                    self._triple_model.subject_id == mid,
                )
                if as_of is not None:
                    stmt_out = stmt_out.where(
                        self._triple_model.valid_from <= as_of,
                        (self._triple_model.valid_to.is_(None)) | (self._triple_model.valid_to >= as_of),
                    )
                else:
                    stmt_out = stmt_out.where(self._triple_model.valid_to.is_(None))
                if filters:
                    stmt_out = stmt_out.where(*filters)
                if predicates:
                    stmt_out = stmt_out.where(self._triple_model.predicate.in_(predicates))
                stmt_out = stmt_out.limit(max_per_source)
                out_edges = list(session.exec(stmt_out).all())

                # Incoming edges to this memory
                stmt_in = select(self._triple_model.subject_id, self._triple_model.predicate).where(
                    self._triple_model.object_id == mid,
                )
                if as_of is not None:
                    stmt_in = stmt_in.where(
                        self._triple_model.valid_from <= as_of,
                        (self._triple_model.valid_to.is_(None)) | (self._triple_model.valid_to >= as_of),
                    )
                else:
                    stmt_in = stmt_in.where(self._triple_model.valid_to.is_(None))
                if filters:
                    stmt_in = stmt_in.where(*filters)
                if predicates:
                    stmt_in = stmt_in.where(self._triple_model.predicate.in_(predicates))
                stmt_in = stmt_in.limit(max_per_source)
                in_edges = list(session.exec(stmt_in).all())

                count = 0
                for connected_id, predicate in out_edges + in_edges:
                    if count >= max_per_source:
                        break
                    if connected_id not in seen and connected_id not in memory_ids:
                        seen.add(connected_id)
                        result.append((connected_id, predicate, mid))
                        count += 1

        return result


__all__ = ["SQLiteTripleRepo"]
