from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import select

from memu.database.models import DossierCandidate
from memu.database.repositories.dossier_candidate import DossierCandidateRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState
from memu.utils.taxonomy import normalize_category_name


class SQLiteDossierCandidateRepo(SQLiteRepoBase, DossierCandidateRepo):
    def __init__(
        self,
        *,
        dossier_candidate_model: type[Any],
        state: DatabaseState,
        sqla_models: SQLiteSQLAModels,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        super().__init__(state=state, sqla_models=sqla_models, sessions=sessions, scope_fields=scope_fields)
        self._candidate_model = dossier_candidate_model

    @staticmethod
    def _to_candidate(row: Any) -> DossierCandidate:
        return DossierCandidate(
            id=row.id,
            proposed_name=row.proposed_name,
            normalized_name=row.normalized_name,
            item_id=row.item_id,
            segment_id=row.segment_id,
            memory_day=row.memory_day,
            resolved_category_id=row.resolved_category_id,
            resolved_at=row.resolved_at,
            last_considered_at=row.last_considered_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def add_candidate(
        self,
        *,
        proposed_name: str,
        item_id: str,
        where: Mapping[str, Any],
        segment_id: str | None = None,
        memory_day: str | None = None,
        session: Any | None = None,
    ) -> DossierCandidate:
        if session is None:
            with self._sessions.session() as managed_session:
                candidate = self.add_candidate(
                    proposed_name=proposed_name,
                    item_id=item_id,
                    where=where,
                    segment_id=segment_id,
                    memory_day=memory_day,
                    session=managed_session,
                )
                managed_session.commit()
                return candidate

        scope = self._require_scope(where)
        normalized_name = normalize_category_name(proposed_name)
        if normalized_name is None:
            raise ValueError("Dossier candidate name must contain letters or digits")
        normalized_day = date.fromisoformat(memory_day).isoformat() if memory_day is not None else None

        candidate = DossierCandidate(
            proposed_name=proposed_name.strip(),
            normalized_name=normalized_name,
            item_id=item_id,
            segment_id=segment_id,
            memory_day=normalized_day,
        )
        statement = sqlite_insert(self._candidate_model).values(**candidate.model_dump(), **scope)
        statement = statement.on_conflict_do_nothing(
            index_elements=[
                *[getattr(self._candidate_model, field) for field in self._scope_fields],
                self._candidate_model.normalized_name,
                self._candidate_model.item_id,
            ]
        )
        session.execute(statement)
        row = session.exec(
            select(self._candidate_model).where(
                self._candidate_model.normalized_name == normalized_name,
                self._candidate_model.item_id == item_id,
                *self._build_filters(self._candidate_model, scope),
            )
        ).one()
        return self._to_candidate(row)

    def list_candidates(
        self,
        where: Mapping[str, Any],
        *,
        unresolved_only: bool = True,
        session: Any | None = None,
    ) -> list[DossierCandidate]:
        scope = self._require_scope(where)
        statement = select(self._candidate_model).where(*self._build_filters(self._candidate_model, scope))
        if unresolved_only:
            statement = statement.where(self._candidate_model.resolved_category_id.is_(None))
        statement = statement.order_by(self._candidate_model.created_at, self._candidate_model.id)
        if session is None:
            with self._sessions.session() as managed_session:
                rows = managed_session.exec(statement).all()
        else:
            rows = session.exec(statement).all()
        return [self._to_candidate(row) for row in rows]

    def mark_candidates_considered(
        self,
        candidate_ids: Sequence[str],
        considered_at: datetime,
        where: Mapping[str, Any],
        session: Any,
    ) -> list[DossierCandidate]:
        scope = self._require_scope(where)
        ids = set(candidate_ids)
        if not ids:
            return []
        rows = session.exec(
            select(self._candidate_model).where(
                self._candidate_model.id.in_(ids),
                *self._build_filters(self._candidate_model, scope),
            )
        ).all()
        found = {row.id for row in rows}
        if found != ids:
            raise KeyError(f"Dossier candidates not found in scope: {sorted(ids - found)}")
        now = self._now()
        for row in rows:
            row.last_considered_at = considered_at
            row.updated_at = now
            session.add(row)
        session.flush()
        return [self._to_candidate(row) for row in sorted(rows, key=lambda value: (value.created_at, value.id))]

    def resolve_candidates(
        self,
        candidate_ids: Sequence[str],
        category_id: str,
        where: Mapping[str, Any],
        session: Any | None = None,
    ) -> list[DossierCandidate]:
        scope = self._require_scope(where)
        ids = set(candidate_ids)
        if not ids:
            return []
        if session is None:
            with self._sessions.session() as managed_session:
                candidates = self.resolve_candidates(list(ids), category_id, scope, session=managed_session)
                managed_session.commit()
                return candidates

        category = session.exec(
            select(self._sqla_models.MemoryCategory).where(
                self._sqla_models.MemoryCategory.id == category_id,
                *self._build_filters(self._sqla_models.MemoryCategory, scope),
            )
        ).first()
        if category is None:
            raise KeyError(f"Category with id {category_id} not found in scope")

        rows = session.exec(
            select(self._candidate_model).where(
                self._candidate_model.id.in_(ids),
                *self._build_filters(self._candidate_model, scope),
            )
        ).all()
        found = {row.id for row in rows}
        if found != ids:
            raise KeyError(f"Dossier candidates not found in scope: {sorted(ids - found)}")
        conflicts = [row.id for row in rows if row.resolved_category_id not in (None, category_id)]
        if conflicts:
            raise ValueError(f"Dossier candidates already resolved to another category: {sorted(conflicts)}")

        now = self._now()
        for row in rows:
            if row.resolved_category_id is None:
                row.resolved_category_id = category_id
                row.resolved_at = now
                row.updated_at = now
                session.add(row)
        session.flush()
        return [self._to_candidate(row) for row in sorted(rows, key=lambda value: (value.created_at, value.id))]


__all__ = ["SQLiteDossierCandidateRepo"]
