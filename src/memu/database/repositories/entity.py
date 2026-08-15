from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from memu.database.models import Entity


@runtime_checkable
class EntityRepo(Protocol):
    """Repository contract for entity records."""

    def get_or_create(
        self,
        name: str,
        entity_type: str,
        user_data: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Entity: ...

    def create(
        self,
        name: str,
        entity_type: str,
        user_data: Mapping[str, Any],
        *,
        properties: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> Entity: ...

    def update(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any],
        name: str | None = None,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
        property_updates: Mapping[str, Any] | None = None,
        property_removals: set[str] | None = None,
        session: Any | None = None,
    ) -> Entity: ...

    def list_all(
        self,
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[Entity]: ...

    def delete(
        self,
        entity_id: str,
        *,
        where: Mapping[str, Any],
        session: Any | None = None,
    ) -> None: ...

    def write_lock(self) -> Any: ...

    def list_by_ids(
        self,
        entity_ids: set[str],
        where: Mapping[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> list[Entity]: ...

    def bind_source_refs(
        self,
        bindings: Mapping[str, str],
        where: Mapping[str, Any] | None = None,
        session: Any | None = None,
    ) -> list[Entity]: ...
