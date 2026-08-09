from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from memu.database.models import CategoryItem


@runtime_checkable
class CategoryItemRepo(Protocol):
    """Repository contract for item/category relations."""

    relations: list[CategoryItem]

    def list_relations(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[CategoryItem]: ...

    def link_item_category(
        self,
        item_id: str,
        category_id: str,
        user_data: dict[str, Any],
        session: Any | None = None,
    ) -> CategoryItem: ...

    def unlink_item_category(
        self,
        item_id: str,
        category_id: str,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> bool: ...

    def refresh_category_relations(
        self,
        category_id: str,
        where: Mapping[str, Any],
    ) -> list[CategoryItem]: ...

    def get_item_categories(self, item_id: str) -> list[CategoryItem]: ...
