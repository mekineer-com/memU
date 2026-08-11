import json
from datetime import datetime

from memu.app.service import MemoryService
from memu.app.memorize_segments import grouped_chat_happened_at


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


def test_resolve_entry_happened_at_uses_episode_provenance_start() -> None:
    service = _service()
    raw_text = json.dumps([
        {"role": "user", "content": "zero", "ts_ms": 1737849600000},
        {"role": "assistant", "content": "one", "ts_ms": 1737849660000},
    ])

    happened_at_map = service._extract_message_happened_at_map(raw_text)

    direct = service._resolve_entry_happened_at([1], happened_at_map)
    fallback = service._resolve_entry_happened_at([], happened_at_map)

    assert direct is not None
    assert fallback is not None
    assert direct == datetime.fromtimestamp(1737849660)
    assert fallback == datetime.fromtimestamp(1737849600)


def test_resolve_source_message_ids_always_uses_episode_provenance() -> None:
    service = _service()
    episode = [0, 1, 2, 3]

    assert service._resolve_source_message_ids(None, episode) == episode
    assert service._resolve_source_message_ids([], episode) == episode
    assert service._resolve_source_message_ids([99], episode) == episode
    assert service._resolve_source_message_ids([1, 2], episode) == episode
    assert service._resolve_source_message_ids([1, 99], episode) == episode
    assert service._resolve_source_message_ids([1, 2], None) == []


def test_resolve_source_message_ids_handles_malformed_input_without_raising() -> None:
    # The resolver is the boundary that prevents stale LLM output from turning
    # generalized memories back into individual-message memories.
    service = _service()
    episode = [0, 1, 2]

    assert service._resolve_source_message_ids("unexpected string", episode) == episode
    assert service._resolve_source_message_ids({"malformed": "dict"}, episode) == episode
    assert service._resolve_source_message_ids(42, episode) == episode
    assert service._resolve_source_message_ids([None, "x", 2.5, {"k": 1}], episode) == episode
