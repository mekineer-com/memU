from __future__ import annotations

from typing import Protocol, runtime_checkable

from memu.database.models import Triple


@runtime_checkable
class TripleRepo(Protocol):
    """Repository contract for triple (edge) records."""

    def add(self, triple: Triple) -> Triple: ...

    def get_edges_from(
        self, subject_id: str, predicate: str | None = None, current_only: bool = True
    ) -> list[Triple]: ...

    def get_edges_to(
        self, object_id: str, predicate: str | None = None, current_only: bool = True
    ) -> list[Triple]: ...

    def invalidate(self, subject_id: str, predicate: str, object_id: str) -> None: ...

    def get_connected_memory_ids(
        self,
        memory_ids: list[str],
        predicates: list[str] | None = None,
        max_per_source: int = 3,
    ) -> list[str]: ...
