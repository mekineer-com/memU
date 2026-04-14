"""SQLite triple repository implementation."""

from __future__ import annotations

from typing import Any

from sqlmodel import select

from memu.database.models import Triple
from memu.database.repositories.triple import TripleRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState


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
            confidence=row.confidence if row.confidence is not None else 1.0,
            source_memory_id=row.source_memory_id,
            properties=row.properties or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def add(self, triple: Triple, session: Any | None = None) -> Triple:
        now = self._now()
        if session is None:
            with self._sessions.session() as db_session:
                persisted = self.add(triple, session=db_session)
                db_session.commit()
                return persisted

        row = self._triple_model(
            id=triple.id,
            subject_id=triple.subject_id,
            subject_kind=triple.subject_kind,
            predicate=triple.predicate,
            object_id=triple.object_id,
            object_kind=triple.object_kind,
            valid_from=triple.valid_from or now,
            valid_to=triple.valid_to,
            confidence=triple.confidence,
            source_memory_id=triple.source_memory_id,
            properties=triple.properties,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        return self._row_to_triple(row)

    def get_edges_from(
        self, subject_id: str, predicate: str | None = None, current_only: bool = True
    ) -> list[Triple]:
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                self._triple_model.subject_id == subject_id
            )
            if predicate is not None:
                stmt = stmt.where(self._triple_model.predicate == predicate)
            if current_only:
                stmt = stmt.where(self._triple_model.valid_to.is_(None))
            rows = session.exec(stmt).all()
            return [self._row_to_triple(r) for r in rows]

    def get_edges_to(
        self, object_id: str, predicate: str | None = None, current_only: bool = True
    ) -> list[Triple]:
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                self._triple_model.object_id == object_id
            )
            if predicate is not None:
                stmt = stmt.where(self._triple_model.predicate == predicate)
            if current_only:
                stmt = stmt.where(self._triple_model.valid_to.is_(None))
            rows = session.exec(stmt).all()
            return [self._row_to_triple(r) for r in rows]

    def invalidate(self, subject_id: str, predicate: str, object_id: str) -> None:
        now = self._now()
        with self._sessions.session() as session:
            stmt = select(self._triple_model).where(
                self._triple_model.subject_id == subject_id,
                self._triple_model.predicate == predicate,
                self._triple_model.object_id == object_id,
                self._triple_model.valid_to.is_(None),
            )
            rows = session.exec(stmt).all()
            for row in rows:
                row.valid_to = now
                row.updated_at = now
                session.add(row)
            session.commit()

    def get_connected_memory_ids(
        self,
        memory_ids: list[str],
        predicates: list[str] | None = None,
        max_per_source: int = 3,
    ) -> list[str]:
        if not memory_ids:
            return []

        seen: set[str] = set()
        result: list[str] = []

        with self._sessions.session() as session:
            for mid in memory_ids:
                # Outgoing edges from this memory
                stmt_out = select(self._triple_model.object_id).where(
                    self._triple_model.subject_id == mid,
                    self._triple_model.valid_to.is_(None),
                )
                if predicates:
                    stmt_out = stmt_out.where(self._triple_model.predicate.in_(predicates))
                stmt_out = stmt_out.limit(max_per_source)
                out_ids = [r for r in session.exec(stmt_out).all()]

                # Incoming edges to this memory
                stmt_in = select(self._triple_model.subject_id).where(
                    self._triple_model.object_id == mid,
                    self._triple_model.valid_to.is_(None),
                )
                if predicates:
                    stmt_in = stmt_in.where(self._triple_model.predicate.in_(predicates))
                stmt_in = stmt_in.limit(max_per_source)
                in_ids = [r for r in session.exec(stmt_in).all()]

                count = 0
                for connected_id in out_ids + in_ids:
                    if count >= max_per_source:
                        break
                    if connected_id not in seen and connected_id not in memory_ids:
                        seen.add(connected_id)
                        result.append(connected_id)
                        count += 1

        return result


__all__ = ["SQLiteTripleRepo"]
