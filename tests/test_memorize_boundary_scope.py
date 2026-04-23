"""Engine boundary assertion: scope missing soul_id must fail loud.

Without this, a caller (server, test, MCP tool) could pass a non-empty
scope like {"user_id": "marcos"} and the engine would quietly operate
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
                user={"user_id": "marcos"},
            )
        )


def test_memorize_rejects_blank_soul_id() -> None:
    service = _service()
    with pytest.raises(ValueError, match="soul_id"):
        asyncio.run(
            service.memorize(
                resource_url="file:///nowhere",
                modality="conversation",
                user={"user_id": "marcos", "soul_id": "   "},
            )
        )


def test_memorize_accepts_empty_scope_dict() -> None:
    # Empty dict is the "no scope provided" case used by tests that don't
    # need isolation — should pass through without raising.
    service = _service()
    try:
        asyncio.run(
            service.memorize(
                resource_url="file:///nowhere",
                modality="conversation",
                user={},
                raw_text="",
            )
        )
    except ValueError as e:
        if "soul_id" in str(e):
            raise AssertionError(f"empty scope should not trigger soul_id assertion: {e}") from e
    except Exception:
        # Other errors (bad resource_url, empty raw_text, etc.) are fine —
        # we're only asserting the soul_id gate doesn't fire on empty dict.
        pass


def test_memorize_accepts_none_scope() -> None:
    # user=None is legitimate (scope-less engine operation).
    service = _service()
    try:
        asyncio.run(
            service.memorize(
                resource_url="file:///nowhere",
                modality="conversation",
                user=None,
                raw_text="",
            )
        )
    except ValueError as e:
        if "soul_id" in str(e):
            raise AssertionError(f"user=None should not trigger soul_id assertion: {e}") from e
    except Exception:
        pass
