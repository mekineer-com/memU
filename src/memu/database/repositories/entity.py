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

    def list_all(self, where: Mapping[str, Any] | None = None) -> list[Entity]: ...
