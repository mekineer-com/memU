from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from memu.database.models import Resource


@runtime_checkable
class ResourceRepo(Protocol):
    """Repository contract for resource records."""

    resources: dict[str, Resource]

    def list_resources(self, where: Mapping[str, Any] | None = None) -> dict[str, Resource]: ...

    def clear_resources(self, where: Mapping[str, Any] | None = None) -> dict[str, Resource]: ...

    def create_resource(
        self,
        *,
        url: str,
        modality: str,
        local_path: str,
        caption: str | None,
        embedding: list[float] | None,
        user_data: dict[str, Any],
        episode_id: str | None = None,
        conversation_id: str | None = None,
        memory_retrieve_history: list[str] | None = None,
        memory_prior_context: list[str] | None = None,
        session: Any | None = None,
    ) -> Resource: ...

