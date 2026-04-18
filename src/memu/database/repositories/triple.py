from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from memu.database.models import Triple


@runtime_checkable
class TripleRepo(Protocol):
    """Repository contract for triple (edge) records."""

    def add(
        self,
        triple: Triple,
        user_data: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Triple: ...

    def get_edges_from(
        self,
        subject_id: str,
        predicate: str | None = None,
        current_only: bool = True,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[Triple]: ...

    def get_edges_to(
        self,
        object_id: str,
        predicate: str | None = None,
        current_only: bool = True,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[Triple]: ...

    def invalidate(self, subject_id: str, predicate: str, object_id: str) -> None: ...

    def get_connected_memory_ids(
        self,
        memory_ids: list[str],
        predicates: list[str] | None = None,
        max_per_source: int = 3,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[str]: ...
