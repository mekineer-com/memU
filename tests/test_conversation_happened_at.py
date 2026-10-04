import json
from datetime import datetime

from memu.app.service import MemoryService
from memu.app.memorize_segments import grouped_chat_happened_at
from memu.utils.conversation import format_grouped_chat_history


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_extract_message_happened_at_map_uses_grouped_chat_timestamp_order() -> None:
    service = _service()
    raw_text = json.dumps([
        {"role": "user", "content": "one", "ts_ms": 1737849600000},
        {"role": "assistant", "content": "two", "timestamp": "2025-01-26T03:40:49.205Z"},
        {"role": "user", "content": "three", "received_at": "2025-01-27T01:02:03Z"},
        {"role": "user", "content": "four", "created_at": "2025-01-28T01:02:03Z"},
    ])

    happened_at_map = service._extract_message_happened_at_map(raw_text)

    assert sorted(happened_at_map) == [0, 1, 2, 3]
    assert happened_at_map[1] is not None
    assert happened_at_map[2] is not None
    assert happened_at_map[3] is not None
    assert happened_at_map[0] == datetime.fromtimestamp(1737849600)
    assert happened_at_map[1] == datetime(2025, 1, 26, 3, 40, 49, 205000)
    assert happened_at_map[2] == datetime(2025, 1, 27, 1, 2, 3)
    assert happened_at_map[3] == datetime(2025, 1, 28, 1, 2, 3)


def test_grouped_chat_happened_at_accepts_timestamp_fallback() -> None:
    happened_at = grouped_chat_happened_at({"timestamp": "2026-01-02T10:00:00-05:00"})

    assert happened_at is not None
    assert happened_at == datetime(2026, 1, 2, 10)


def test_grouped_chat_happened_at_preserves_calendar_only_day() -> None:
    assert grouped_chat_happened_at({"received_at": "2026-01-02"}) == datetime(2026, 1, 2)


def test_imported_source_day_and_app_label_survive_utc_normalization() -> None:
    row = {"conversation_id": "import:dm:fictional-chat", "app_label": "Nomi",
           "source_day": "2025-02-01", "received_at": "2025-01-31T22:30:00+00:00",
           "role": "user", "name": "TestSpeaker", "content": "hello"}
    assert grouped_chat_happened_at(row) == datetime(2025, 2, 1)
    rendered = format_grouped_chat_history([row], time_label_resolver=lambda value: value)
    assert "My Nomi Conversations:" in rendered and "--- 2025-02-01 ---" in rendered
    assert "[TestSpeaker] hello" in rendered and "My SillyTavern" not in rendered


def test_resolve_entry_happened_at_uses_memory_selected_day() -> None:
    service = _service()
    happened_at_map = {
        "2025-01-25": datetime.fromtimestamp(1737849600),
        "2025-01-26": datetime.fromtimestamp(1737936000),
    }

    assert service._resolve_entry_happened_at("2025-01-26", happened_at_map) == datetime.fromtimestamp(1737936000)
    assert service._resolve_entry_happened_at(None, happened_at_map) is None
