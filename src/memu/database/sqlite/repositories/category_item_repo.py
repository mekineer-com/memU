"""SQLite category-item relation repository implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sqlmodel import select

from memu.database.models import CategoryItem
from memu.database.repositories.category_item import CategoryItemRepo
from memu.database.sqlite.repositories.base import SQLiteRepoBase
from memu.database.sqlite.schema import SQLiteSQLAModels
from memu.database.sqlite.session import SQLiteSessionManager
from memu.database.state import DatabaseState

logger = logging.getLogger(__name__)


class SQLiteCategoryItemRepo(SQLiteRepoBase, CategoryItemRepo):
    """SQLite implementation of category-item relation repository."""

    def __init__(
        self,
        *,
        state: DatabaseState,
        category_item_model: type[Any],
        sqla_models: SQLiteSQLAModels,
        sessions: SQLiteSessionManager,
        scope_fields: list[str],
    ) -> None:
        """Initialize category-item repository.

        Args:
            state: Shared database state for caching.
            category_item_model: SQLModel class for category-item relations.
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
        self._category_item_model = category_item_model
        self.relations = self._state.relations

    def list_relations(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[CategoryItem]:
        """List category-item relations matching the where clause.

        Args:
            where: Optional filter conditions.

        Returns:
            List of CategoryItem relations.
        """
        stmt = select(self._category_item_model)
        filters = self._build_filters(self._category_item_model, where)
        if filters:
            stmt = stmt.where(*filters)
        if session is None:
            with self._sessions.session() as managed_session:
                rows = managed_session.exec(stmt).all()
        else:
            rows = session.exec(stmt).all()

        result: list[CategoryItem] = []
        for row in rows:
            rel = CategoryItem(
                id=row.id,
                item_id=row.item_id,
                category_id=row.category_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            result.append(rel)
            if session is None and not any(r.id == rel.id for r in self.relations):
                self.relations.append(rel)

        return result

    def link_item_category(
        self,
        item_id: str,
        category_id: str,
        user_data: dict[str, Any],
        session: Any | None = None,
    ) -> CategoryItem:
        """Create a link between an item and a category.

        Args:
            item_id: Memory item ID.
            category_id: Category ID.
            user_data: User scope data.

        Returns:
            Created CategoryItem relation.
        """
        if session is None:
            with self._sessions.session() as session:
                rel = self.link_item_category(
                    item_id=item_id,
                    category_id=category_id,
                    user_data=user_data,
                    session=session,
                )
                session.commit()
                if not any(existing.id == rel.id for existing in self.relations):
                    self.relations.append(rel)
                return rel

        # Check if relation already exists
        where: dict[str, Any] = {
            "item_id": item_id,
            "category_id": category_id,
            **user_data,
        }
        stmt = select(self._category_item_model)
        filters = self._build_filters(self._category_item_model, where)
        if filters:
            stmt = stmt.where(*filters)
        existing = session.exec(stmt).first()

        if existing:
            rel = CategoryItem(
                id=existing.id,
                item_id=existing.item_id,
                category_id=existing.category_id,
                created_at=existing.created_at,
                updated_at=existing.updated_at,
            )
            return rel

        # Create new relation
        now = self._now()
        row = self._category_item_model(
            item_id=item_id,
            category_id=category_id,
            created_at=now,
            updated_at=now,
            **user_data,
        )
        session.add(row)
        session.flush()
        session.refresh(row)

        rel = CategoryItem(
            id=row.id,
            item_id=row.item_id,
            category_id=row.category_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            **user_data,
        )
        return rel

    def unlink_item_category(
        self,
        item_id: str,
        category_id: str,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> bool:
        """Remove a link between an item and a category.

        Args:
            item_id: Memory item ID.
            category_id: Category ID.
        """
        if session is None:
            with self._sessions.session() as managed_session:
                deleted = self.unlink_item_category(
                    item_id,
                    category_id,
                    where,
                    session=managed_session,
                )
                managed_session.commit()
            if deleted:
                self.relations[:] = [
                    relation
                    for relation in self.relations
                    if not (relation.item_id == item_id and relation.category_id == category_id)
                ]
            return deleted

        filters = self._build_filters(
            self._category_item_model,
            {"item_id": item_id, "category_id": category_id, **dict(where or {})},
        )
        row = session.exec(select(self._category_item_model).where(*filters)).first()
        if row:
            session.delete(row)
            session.flush()
            return True
        return False

    def refresh_category_relations(
        self,
        category_id: str,
        where: Mapping[str, Any],
    ) -> list[CategoryItem]:
        relations = [
            relation
            for relation in self.list_relations(where)
            if relation.category_id == category_id
        ]
        self.relations[:] = [
            relation for relation in self.relations if relation.category_id != category_id
        ]
        self.relations.extend(relations)
        return relations

    def get_item_categories(self, item_id: str) -> list[CategoryItem]:
        """Get all category relations for a given item.

        Args:
            item_id: Memory item ID.

        Returns:
            List of CategoryItem relations for the item.
        """
        return self.list_relations({"item_id": item_id})


__all__ = ["SQLiteCategoryItemRepo"]
