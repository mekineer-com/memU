import json

from memu.app.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(database_config={"metadata_store": {"provider": "sqlite", "dsn": "sqlite:///:memory:"}})


def test_extract_message_happened_at_map_prefers_ts_ms_and_falls_back() -> None:
    service = _service()
    raw_text = json.dumps([
        {"role": "user", "content": "one", "ts_ms": 1737849600000},
        {"role": "assistant", "content": "two", "timestamp": "2025-01-26T03:40:49.205Z"},
        {"role": "user", "content": "three", "created_at": "2025-01-27T01:02:03Z"},
    ])

    happened_at_map = service._extract_message_happened_at_map(raw_text)

    assert sorted(happened_at_map) == [0, 1, 2]
    assert happened_at_map[1] is not None
    assert happened_at_map[2] is not None
    assert happened_at_map[0].to_iso8601_string() == "2025-01-26T00:00:00Z"
    assert happened_at_map[1].to_iso8601_string() == "2025-01-26T00:00:00Z"
    assert happened_at_map[2].to_iso8601_string() == "2025-01-27T00:00:00Z"


def test_resolve_entry_happened_at_uses_source_message_ids_then_episode_fallback() -> None:
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
    assert direct.to_iso8601_string() == "2025-01-26T00:00:00Z"
    assert fallback.to_iso8601_string() == "2025-01-26T00:00:00Z"


def test_resolve_source_message_ids_falls_back_to_episode_when_model_emits_nothing() -> None:
    service = _service()
    episode = [0, 1, 2, 3]

    assert service._resolve_source_message_ids(None, episode) == episode
    assert service._resolve_source_message_ids([], episode) == episode
    assert service._resolve_source_message_ids([99], episode) == episode  # all out-of-range → fallback
    assert service._resolve_source_message_ids([1, 2], episode) == [1, 2]  # valid subset kept
    assert service._resolve_source_message_ids([1, 99], episode) == [1]  # keep valid, drop invalid
    assert service._resolve_source_message_ids([1, 2], None) == [1, 2]  # no episode → pass through


def test_resolve_source_message_ids_handles_malformed_input_without_raising() -> None:
    # The narrowing guard at memorize.py:1651 was removed; the resolver is
    # now the single normalization boundary for anything the LLM might emit.
    service = _service()
    episode = [0, 1, 2]

    # Strings, dicts, scalars, mixed garbage — all coerce to "empty emitted",
    # falling back to the full episode range.
    assert service._resolve_source_message_ids("unexpected string", episode) == episode
    assert service._resolve_source_message_ids({"malformed": "dict"}, episode) == episode
    assert service._resolve_source_message_ids(42, episode) == episode  # not iterable → parsed empty
    assert service._resolve_source_message_ids([None, "x", 2.5, {"k": 1}], episode) == [2]
