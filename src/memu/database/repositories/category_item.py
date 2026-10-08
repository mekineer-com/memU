from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from memu.database.models import CategoryItem


@runtime_checkable
class CategoryItemRepo(Protocol):
    """Repository contract for item/category relations."""

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

    def mark_reviewed(
        self,
        category_id: str,
        item_ids: set[str],
        where: Mapping[str, Any],
        *,
        reviewed_at: datetime,
        session: Any,
    ) -> None: ...
