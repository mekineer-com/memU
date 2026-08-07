from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import EllipsisType
from typing import Any, Literal, Protocol, runtime_checkable

from memu.database.models import DossierKind, MemoryCategory


@runtime_checkable
class MemoryCategoryRepo(Protocol):
    """Repository contract for memory categories."""

    categories: dict[str, MemoryCategory]

    def list_categories(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> dict[str, MemoryCategory]: ...

    def list_anchor_categories(self, where: Mapping[str, Any]) -> dict[str, MemoryCategory]: ...

    def list_categories_by_activity(
        self,
        where: Mapping[str, Any],
        *,
        kind: DossierKind,
    ) -> list[MemoryCategory]: ...

    def clear_categories(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryCategory]: ...

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
    ) -> MemoryCategory: ...

    def create_category_strict(
        self,
        *,
        name: str,
        description: str,
        embedding: list[float],
        user_data: dict[str, Any],
        kind: DossierKind,
        session: Any,
    ) -> MemoryCategory: ...

    def update_category(
        self,
        *,
        category_id: str,
        name: str | None = None,
        description: str | None = None,
        embedding: list[float] | None = None,
        summary: str | None = None,
        previous_summary: str | None = None,
        kind: DossierKind | None | EllipsisType = ...,
        lore_subtype: str | None | EllipsisType = ...,
        entity_id: str | None | EllipsisType = ...,
        anchor_role: Literal["soul", "user"] | None | EllipsisType = ...,
        last_evidence_at: datetime | None | EllipsisType = ...,
        last_revised_at: datetime | None | EllipsisType = ...,
    ) -> MemoryCategory: ...

    def approve_category_summary(
        self,
        category_id: str,
        where: Mapping[str, Any] | None = None,
    ) -> MemoryCategory: ...
