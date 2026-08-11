from memu.app.memorize import StructuredMemoryEntry
from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_prune_keeps_profiles_and_non_speech_events() -> None:
    service = _service()
    entries = [
        StructuredMemoryEntry("profile", "Marcos has PTSD from past abuse.", ["profiles"], "user", 0.9, [], None),
        StructuredMemoryEntry(
            "knowledge",
            "Marcos is taking ashwagandha to help with PTSD symptoms.",
            ["health"],
            "user",
            0.9,
            [2, 4],
            0.6,
        ),
    ]

    kept = service._prune_extracted_entry_duplicates(entries)

    assert len(kept) == 2
    assert kept[0].memory_type == "profile"
    assert kept[1].memory_type == "knowledge"


def test_prune_keeps_same_words_from_different_speakers() -> None:
    first = StructuredMemoryEntry(
        "social",
        "The shared plan feels right.",
        [],
        "group_member",
        0.8,
        [0],
        None,
        speaker_id="entity:speaker-a",
    )
    second = first._replace(speaker_id="entity:speaker-b", source_message_ids=[1])
    duplicate = first._replace(source_message_ids=[2])

    kept = _service()._prune_extracted_entry_duplicates([first, second, duplicate])

    assert [entry.speaker_id for entry in kept] == ["entity:speaker-a", "entity:speaker-b"]
