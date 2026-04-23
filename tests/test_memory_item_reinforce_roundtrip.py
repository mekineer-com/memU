"""Contract test for the sqlite reinforce branch.

History: `create_item` early-returns into `create_item_reinforce` when
`reinforce=True` (the default for extracted memories). That call site was
silently dropping `source_message_ids` and `reflection_salience`, so every
live-path memory stored null for both fields regardless of what memorize
computed. Commit 39511ef fixed the drop. This test pins the fix.

Without this guard, a regression would go undetected until someone noticed
speaker attribution stopped working — which, as the session history shows,
takes months.
"""

from __future__ import annotations

from pydantic import BaseModel

from memu.app.service import MemoryService


class ReinforceScope(BaseModel):
    user_id: str | None = None


def _service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": ReinforceScope},
    )


def test_create_item_with_reinforce_persists_source_message_ids_and_reflection_salience() -> None:
    service = _service()
    store = service._get_database()
    user_data = {"user_id": "alice"}

    item = store.memory_item_repo.create_item(
        memory_type="event",
        summary="phase1 reinforce provenance roundtrip",
        embedding=[0.1, 0.2, 0.3],
        user_data=user_data,
        reinforce=True,
        source_role="user",
        speaker_id="user:alice",
        speaker_label="Alice",
        source_message_ids=[4, 5, 6],
        reflection_salience=0.72,
    )

    loaded = store.memory_item_repo.get_item(item.id)
    assert loaded is not None
    assert loaded.source_message_ids == [4, 5, 6]
    assert loaded.reflection_salience == 0.72
    assert loaded.speaker_id == "user:alice"


def test_reinforce_second_call_with_same_summary_updates_fields_from_existing() -> None:
    service = _service()
    store = service._get_database()
    user_data = {"user_id": "alice"}

    first = store.memory_item_repo.create_item(
        memory_type="event",
        summary="dedupe me on content hash",
        embedding=[0.1, 0.2, 0.3],
        user_data=user_data,
        reinforce=True,
        source_role="user",
        source_message_ids=[0, 1],
        reflection_salience=0.5,
    )

    second = store.memory_item_repo.create_item(
        memory_type="event",
        summary="dedupe me on content hash",
        embedding=[0.1, 0.2, 0.3],
        user_data=user_data,
        reinforce=True,
        source_role="user",
        source_message_ids=[7, 8, 9],
        reflection_salience=0.9,
    )

    assert second.id == first.id, "reinforce should return the same row id"
    loaded = store.memory_item_repo.get_item(first.id)
    assert loaded is not None
    assert loaded.source_message_ids == [7, 8, 9]
    assert loaded.reflection_salience == 0.9


def test_reinforce_does_not_clobber_existing_fields_with_none() -> None:
    service = _service()
    store = service._get_database()
    user_data = {"user_id": "alice"}

    first = store.memory_item_repo.create_item(
        memory_type="event",
        summary="preserve on None reinforce",
        embedding=[0.1, 0.2, 0.3],
        user_data=user_data,
        reinforce=True,
        source_role="user",
        source_message_ids=[3, 4],
        reflection_salience=0.6,
    )

    second = store.memory_item_repo.create_item(
        memory_type="event",
        summary="preserve on None reinforce",
        embedding=[0.1, 0.2, 0.3],
        user_data=user_data,
        reinforce=True,
        source_role="user",
    )

    assert second.id == first.id
    loaded = store.memory_item_repo.get_item(first.id)
    assert loaded is not None
    assert loaded.source_message_ids == [3, 4]
    assert loaded.reflection_salience == 0.6
