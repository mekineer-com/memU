from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from memu.database.models import MemoryItem, MemoryType


@runtime_checkable
class MemoryItemRepo(Protocol):
    """Repository contract for memory items."""

    items: dict[str, MemoryItem]

    def get_item(self, item_id: str, *, include_superseded: bool = False) -> MemoryItem | None: ...

    def list_items(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        include_superseded: bool = False,
    ) -> dict[str, MemoryItem]: ...

    def clear_items(self, where: Mapping[str, Any] | None = None) -> dict[str, MemoryItem]: ...

    def create_item(
        self,
        *,
        resource_id: str | None = None,
        memory_type: MemoryType,
        summary: str,
        embedding: list[float],
        user_data: dict[str, Any],
        tool_record: dict[str, Any] | None = None,
        source_role: str | None = None,
        speaker_id: str | None = None,
        speaker_label: str | None = None,
        confidence: float | None = None,
        source_message_ids: list[int] | None = None,
        happened_at: datetime | None = None,
        reflection_salience: float | None = None,
        emotional_intensity: float | None = None,
        conversation_id: str | None = None,
        episode_id: str | None = None,
        unresolved: str | None = None,
    ) -> MemoryItem: ...

    def update_item(
        self,
        *,
        item_id: str,
        memory_type: MemoryType | None = None,
        summary: str | None = None,
        embedding: list[float] | None = None,
        extra: dict[str, Any] | None = None,
        tool_record: dict[str, Any] | None = None,
        merged_into: str | None = None,
        unresolved: str | None = None,
        session: Any | None = None,
    ) -> MemoryItem: ...

    def delete_item(self, item_id: str) -> None: ...

    def list_items_by_ref_ids(
        self, ref_ids: list[str], where: Mapping[str, Any] | None = None
    ) -> dict[str, MemoryItem]: ...

    def vector_search_items(
        self,
        query_vec: list[float],
        top_k: int,
        where: Mapping[str, Any] | None = None,
        *,
        ranking: str = "similarity",
        recency_decay_days: float = 30.0,
        fts_query: str | None = None,
        fts_enabled: bool = False,
        fts_top_k: int = 20,
        rrf_k: int = 60,
        include_superseded: bool = False,
    ) -> list[tuple[str, float]]: ...

    def load_existing(self) -> None: ...
