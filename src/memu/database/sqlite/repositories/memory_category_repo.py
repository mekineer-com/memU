"""SQLite memory category repository implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from types import EllipsisType
from typing import Any, Literal

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import delete, select

from memu.database.models import DossierKind, MemoryCategory
from memu.database.repositories.memory_category import MemoryCategoryRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState
from memu.utils.taxonomy import DOSSIER_KINDS

logger = logging.getLogger(__name__)


class SQLiteMemoryCategoryRepo(SQLiteRepoBase, MemoryCategoryRepo):
    """SQLite implementation of memory category repository."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        memory_category_model: type[Any],
        sqla_models: SQLiteSQLAModels,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        """Initialize memory category repository.

        Args:
            state: Shared database state for caching.
            memory_category_model: SQLModel class for memory categories.
            sqla_models: SQLAlchemy model container.
            sessions: Session manager for database connections.
            scope_fields: List of user scope field names.
        """
        super().__init__(
            state=state,
            sqla_models=sqla_models,
            sessions=sessions,
            scope_fields=scope_fields,
        )
        self._memory_category_model = memory_category_model
        self.categories = self._state.categories

    def _to_category(self, row: Any) -> MemoryCategory:
        return MemoryCategory(
            id=row.id,
            name=row.name,
            description=row.description,
            embedding=self._normalize_embedding(self._get_row_embedding(row)),
            summary=row.summary,
            previous_description=getattr(row, "previous_description", None),
            approved_description=getattr(row, "approved_description", None),
            previous_summary=row.previous_summary,
            approved_summary=getattr(row, "approved_summary", None),
            kind=getattr(row, "kind", None),
            lore_subtype=getattr(row, "lore_subtype", None),
            entity_id=getattr(row, "entity_id", None),
            anchor_role=getattr(row, "anchor_role", None),
            last_evidence_at=getattr(row, "last_evidence_at", None),
            last_revised_at=getattr(row, "last_revised_at", None),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def list_categories(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> dict[str, MemoryCategory]:
        """List categories matching the where clause.

        Args:
            where: Complete user and soul scope.

        Returns:
            Dictionary of category ID to MemoryCategory mapping.
        """
        stmt = select(self._memory_category_model)
        filters = self._build_filters(self._memory_category_model, where)
        if filters:
            stmt = stmt.where(*filters)
        if session is None:
            with self._sessions.session() as managed_session:
                rows = managed_session.exec(stmt).all()
        else:
            rows = session.exec(stmt).all()

        result: dict[str, MemoryCategory] = {}
        for row in rows:
            cat = self._to_category(row)
            result[row.id] = cat
            if session is None:
                self.categories[row.id] = cat

        return result

    def create_category_strict(
        self,
        *,
        name: str,
        description: str,
        embedding: list[float],
        user_data: dict[str, Any],
        kind: DossierKind,
        session: Any,
    ) -> MemoryCategory:
        scope = self._require_scope(user_data)
        if not name.strip() or not description.strip():
            raise ValueError("Category name and description are required")
        if kind not in DOSSIER_KINDS:
            raise ValueError(f"Invalid dossier kind: {kind}")
        now = self._now()
        row = self._memory_category_model(
            name=name.strip(),
            description=description.strip(),
            embedding=None,
            summary=None,
            kind=kind,
            created_at=now,
            updated_at=now,
            **scope,
        )
        self._set_row_embedding(row, embedding)
        session.add(row)
        session.flush()
        session.refresh(row)
        return self._to_category(row)

    def list_anchor_categories(self, where: Mapping[str, Any]) -> dict[str, MemoryCategory]:
        scope = self._require_scope(where)
        with self._sessions.session() as session:
            stmt = select(self._memory_category_model).where(
                *self._build_filters(self._memory_category_model, scope),
                self._memory_category_model.anchor_role.is_not(None),
            )
            rows = session.exec(stmt).all()
        return {row.anchor_role: self._to_category(row) for row in rows}

    def list_categories_by_activity(
        self,
        where: Mapping[str, Any],
        *,
        kind: DossierKind,
    ) -> list[MemoryCategory]:
        scope = self._require_scope(where)
        with self._sessions.session() as session:
            stmt = (
                select(self._memory_category_model)
                .where(
                    *self._build_filters(self._memory_category_model, scope),
                    self._memory_category_model.kind == kind,
                    self._memory_category_model.anchor_role.is_(None),
                )
                .order_by(
                    self._memory_category_model.last_evidence_at.desc().nulls_last(),
                    func.lower(self._memory_category_model.name),
                    self._memory_category_model.id,
                )
            )
            rows = session.exec(stmt).all()
        return [self._to_category(row) for row in rows]

    def clear_categories(self, where: Mapping[str, Any]) -> dict[str, MemoryCategory]:
        """Clear scoped taxonomy state matching the where clause.

        Args:
            where: Optional filter conditions.

        Returns:
            Dictionary of deleted category ID to MemoryCategory mapping.
        """
        scope = self._require_scope(where)
        filters = self._build_filters(self._memory_category_model, scope)
        with self._sessions.session() as session:
            stmt = select(self._memory_category_model)
            if filters:
                stmt = stmt.where(*filters)
            rows = session.exec(stmt).all()
            deleted = {row.id: self._to_category(row) for row in rows}

            for model in (self._sqla_models.CategoryItem, self._sqla_models.DossierCandidate):
                del_stmt = delete(model)
                related_filters = self._build_filters(model, scope)
                if related_filters:
                    del_stmt = del_stmt.where(*related_filters)
                session.exec(del_stmt)
            del_stmt = delete(self._memory_category_model)
            if filters:
                del_stmt = del_stmt.where(*filters)
            session.exec(del_stmt)
            session.commit()

            for cat_id in deleted:
                self.categories.pop(cat_id, None)
            deleted_ids = set(deleted)
            self._state.relations[:] = [
                relation for relation in self._state.relations if relation.category_id not in deleted_ids
            ]

        return deleted

    def get_or_create_category(
        self,
        *,
        name: str,
        description: str,
        embedding: list[float],
        user_data: dict[str, Any],
        kind: DossierKind | None = None,
        lore_subtype: str | None = None,
        entity_id: str | None = None,
        anchor_role: Literal["soul", "user"] | None = None,
        last_evidence_at: datetime | None = None,
        last_revised_at: datetime | None = None,
        session: Any | None = None,
    ) -> MemoryCategory:
        """Get existing category by name or create a new one.

        Args:
            name: Category name.
            description: Category description.
            embedding: Embedding vector.
            user_data: User scope data.

        Returns:
            Existing or newly created MemoryCategory.
        """
        # Check for existing category with same name and scope
        where: dict[str, Any] = {"name": name, **user_data}
        if session is None:
            with self._sessions.session() as session:
                cat = self.get_or_create_category(
                    name=name,
                    description=description,
                    embedding=embedding,
                    user_data=user_data,
                    kind=kind,
                    lore_subtype=lore_subtype,
                    entity_id=entity_id,
                    anchor_role=anchor_role,
                    last_evidence_at=last_evidence_at,
                    last_revised_at=last_revised_at,
                    session=session,
                )
                session.commit()
                return cat

        with session.no_autoflush:
            stmt = select(self._memory_category_model)
            filters = self._build_filters(self._memory_category_model, where)
            if filters:
                stmt = stmt.where(*filters)
            existing = session.exec(stmt).first()

            if existing:
                cat = self._to_category(existing)
                self.categories[existing.id] = cat
                return cat

            # Create new category
            now = self._now()
            row = self._memory_category_model(
                name=name,
                description=description,
                embedding=None,
                summary=None,
                kind=kind,
                lore_subtype=lore_subtype,
                entity_id=entity_id,
                anchor_role=anchor_role,
                last_evidence_at=last_evidence_at,
                last_revised_at=last_revised_at,
                created_at=now,
                updated_at=now,
                **user_data,
            )
            self._set_row_embedding(row, embedding)
            try:
                with session.begin_nested():
                    session.add(row)
                    session.flush()
                session.refresh(row)
            except IntegrityError:
                existing = session.exec(stmt).first()
                if existing is None:
                    raise
                cat = self._to_category(existing)
                self.categories[existing.id] = cat
                return cat

        cat = self._to_category(row)
        self.categories[row.id] = cat
        return cat

    def approve_category_summary(
        self,
        category_id: str,
        where: Mapping[str, Any] | None = None,
    ) -> MemoryCategory:
        with self._sessions.session() as session:
            filters = [self._memory_category_model.id == category_id, *self._build_filters(self._memory_category_model, where)]
            row = session.exec(select(self._memory_category_model).where(*filters)).first()
            if row is None:
                msg = f"Category with id {category_id} not found"
                raise KeyError(msg)
            row.approved_description = row.description
            row.approved_summary = row.summary
            session.add(row)
            session.commit()
            session.refresh(row)
        cat = self._to_category(row)
        self.categories[row.id] = cat
        return cat

    def update_category(
        self,
        *,
        category_id: str,
        name: str | None = None,
        description: str | None = None,
        embedding: list[float] | None = None,
        summary: str | None = None,
        previous_description: str | None = None,
        previous_summary: str | None = None,
        kind: DossierKind | None | EllipsisType = ...,
        lore_subtype: str | None | EllipsisType = ...,
        entity_id: str | None | EllipsisType = ...,
        anchor_role: Literal["soul", "user"] | None | EllipsisType = ...,
        last_evidence_at: datetime | None | EllipsisType = ...,
        last_revised_at: datetime | None | EllipsisType = ...,
        where: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> MemoryCategory:
        """Update an existing category.

        Args:
            category_id: ID of category to update.
            name: New name (optional).
            description: New description (optional).
            embedding: New embedding vector (optional).
            summary: New summary text (optional).

        Returns:
            Updated MemoryCategory object.

        Raises:
            KeyError: If category not found.
        """
        if session is None:
            with self._sessions.session() as managed_session:
                category = self.update_category(
                    category_id=category_id,
                    name=name,
                    description=description,
                    embedding=embedding,
                    summary=summary,
                    previous_description=previous_description,
                    previous_summary=previous_summary,
                    kind=kind,
                    lore_subtype=lore_subtype,
                    entity_id=entity_id,
                    anchor_role=anchor_role,
                    last_evidence_at=last_evidence_at,
                    last_revised_at=last_revised_at,
                    where=where,
                    session=managed_session,
                )
                managed_session.commit()
            self.categories[category.id] = category
            return category

        filters = [
            self._memory_category_model.id == category_id,
            *self._build_filters(self._memory_category_model, where),
        ]
        row = session.exec(select(self._memory_category_model).where(*filters)).first()

        if row is None:
            msg = f"Category with id {category_id} not found"
            raise KeyError(msg)

        if name is not None:
            row.name = name
        if description is not None:
            row.description = description
        if embedding is not None:
            self._set_row_embedding(row, embedding)
        if summary is not None:
            row.summary = summary
        if previous_description is not None:
            row.previous_description = previous_description
        if previous_summary is not None:
            row.previous_summary = previous_summary
        for field, value in (
            ("kind", kind),
            ("lore_subtype", lore_subtype),
            ("entity_id", entity_id),
            ("anchor_role", anchor_role),
            ("last_evidence_at", last_evidence_at),
            ("last_revised_at", last_revised_at),
        ):
            if value is not ...:
                setattr(row, field, value)
        row.updated_at = self._now()

        session.add(row)
        session.flush()
        session.refresh(row)

        return self._to_category(row)


__all__ = ["SQLiteMemoryCategoryRepo"]
