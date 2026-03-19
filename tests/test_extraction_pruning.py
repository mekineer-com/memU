from memu.app.memorize import StructuredMemoryEntry
from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_prune_keeps_profiles_and_non_speech_events() -> None:
    service = _service()
    entries = [
        StructuredMemoryEntry("profile", "Marcos has PTSD from past abuse.", ["profiles"], "user", 0.9, [], None),
        StructuredMemoryEntry(
            "event",
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
    assert kept[1].memory_type == "event"


def test_prune_drops_speech_act_event_when_profile_already_covers_it() -> None:
    service = _service()
    entries = [
        StructuredMemoryEntry("profile", "Marcos feels lonely most of his life.", ["profiles"], "user", 0.9, [], None),
        StructuredMemoryEntry(
            "event",
            "Marcos expressed that he feels lonely most of his life.",
            ["experiences"],
            "user",
            0.9,
            [0],
            0.7,
        ),
    ]

    kept = service._prune_extracted_entry_duplicates(entries)

    assert len(kept) == 1
    assert kept[0].memory_type == "profile"
