"""Engine boundary assertion: scope missing soul_id must fail loud.

Without this, a caller (server, test, MCP tool) could pass a non-empty
scope like {"user_id": "TestOwner"} and the engine would quietly operate
without soul isolation — memories for one soul could leak into another's
filters. The server always sends soul_id; this assertion catches
accidental regression there.
"""

from __future__ import annotations

import asyncio

import pytest

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
    )


def test_memorize_rejects_non_empty_scope_without_soul_id() -> None:
    service = _service()
    with pytest.raises(ValueError, match="soul_id"):
        asyncio.run(
            service.memorize(
                resource_url="file:///nowhere",
                modality="conversation",
                user={"user_id": "TestOwner"},
            )
        )


def test_memorize_rejects_blank_soul_id() -> None:
    service = _service()
    with pytest.raises(ValueError, match="soul_id"):
        asyncio.run(
            service.memorize(
                resource_url="file:///nowhere",
                modality="conversation",
                user={"user_id": "TestOwner", "soul_id": "   "},
            )
        )
