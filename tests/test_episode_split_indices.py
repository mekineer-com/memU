from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_prepare_episode_uses_message_indices_when_text_missing() -> None:
    service = _service()

    episode_text, message_indices = service._prepare_episode(
        modality="conversation",
        text=None,
        message_indices=["0", "x", 1, 2.0, -1],
    )

    assert episode_text is None
    assert message_indices == [0, 1, 2]
