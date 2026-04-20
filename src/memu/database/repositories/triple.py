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

    def invalidate(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        scope: Mapping[str, Any] | None = None,
    ) -> None: ...

    def get_connected_memory_edges(
        self,
        memory_ids: list[str],
        predicates: list[str] | None = None,
        max_per_source: int = 3,
        where: Mapping[str, Any] | None = None,
        as_of: datetime | None = None,
    ) -> list[tuple[str, str, str]]:
        """Return ``(connected_id, predicate, seed_id)`` tuples for memories
        linked to any of ``memory_ids`` via one of ``predicates``.  seed_id
        is the source memory in ``memory_ids`` that this edge attaches to.
        """
        ...

    def query_entity(
        self,
        entity_id: str,
        as_of: datetime | None = None,
        direction: str = "outgoing",
        where: Mapping[str, Any] | None = None,
    ) -> list[Triple]:
        """Return triples where entity_id is subject (direction='outgoing'), object
        (direction='incoming'), or either (direction='both').

        When as_of is provided, only triples valid at that instant are returned
        (valid_from <= as_of AND (valid_to IS NULL OR valid_to >= as_of)).
        current_only is implicitly overridden by as_of when as_of is set.
        """
        ...

    def timeline(
        self,
        entity_id: str,
        limit: int = 100,
        as_of: datetime | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[Triple]:
        """Return triples where entity_id is subject OR object, ordered by
        valid_from ASC (NULLs last).  limit caps the result set.

        When as_of is provided, only triples valid at that instant are returned.
        """
        ...
