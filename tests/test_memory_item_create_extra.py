from __future__ import annotations

from pydantic import BaseModel

from memu.app.service import MemoryService


class _Scope(BaseModel):
    user_id: str


def test_create_item_accepts_extra_and_merges_tool_record() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": _Scope},
    )
    store = service._get_database()

    item = store.memory_item_repo.create_item(
        memory_type="behavior",
        summary="Marcos prefers practical tooling.",
        embedding=[0.1, 0.2, 0.3],
        user_data={"user_id": "u1"},
        extra={"model": "claude-opus-4-6", "episode_item_title": "Story"},
        tool_record={"when_to_use": "When selecting an implementation path"},
    )

    loaded = store.memory_item_repo.get_item(item.id)
    assert loaded is not None
    assert loaded.extra["model"] == "claude-opus-4-6"
    assert loaded.extra["episode_item_title"] == "Story"
    assert loaded.extra["when_to_use"] == "When selecting an implementation path"
