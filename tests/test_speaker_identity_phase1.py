from __future__ import annotations

from pydantic import BaseModel

from memu.app.service import MemoryService


class SpeakerPhase1Scope(BaseModel):
    user_id: str | None = None


def test_create_item_round_trips_speaker_fields() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}},
        user_config={"model": SpeakerPhase1Scope},
    )
    store = service._get_database()
    item = store.memory_item_repo.create_item(
        memory_type="behavior",
        summary="phase1 speaker fields roundtrip",
        embedding=[0.2, 0.4, 0.6],
        user_data={"user_id": "speaker_phase1"},
        source_role="user",
        speaker_id="user:speaker_phase1",
        speaker_label="Speaker Phase1",
    )

    loaded = store.memory_item_repo.get_item(item.id)
    assert loaded is not None
    assert loaded.speaker_id == "user:speaker_phase1"
    assert loaded.speaker_label == "Speaker Phase1"
